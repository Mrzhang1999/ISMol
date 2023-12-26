import warnings
import torch
from evaluate import metrics_utils
from models import predictor
from models import utils
from models.utils import logger_itm, logger_mlm, FocalLoss, logger_fpp, \
    compute_MoleculeNet_classify
from models.vit import createVisualModel
from transformers.models.bert.modeling_bert import BertConfig
from models.med import BertCrossLayer
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer
warnings.filterwarnings("ignore")

# 模型
class myCLIP(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                # image encoder initialize
                createVisualModel(config['image_size'])
                # Smiles encoder initialize
                AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
            torch.distributed.barrier()

        # Visual encoder initialize
        self.visual_encoder = createVisualModel()

        self.tokenizer = self.init_tokenizer()
        self.vocab = self.tokenizer.vocab
        # smiles encoder initialize
        self.smiles_encoder = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        self.smiles_encoder.resize_token_embeddings(config['input_smiles_embed_size'])

        self.cross_modal_smiles_transform = nn.Linear(config['input_smiles_embed_size'], config['hidden_size'])
        self.cross_modal_smiles_transform.apply(utils.init_weights)
        self.cross_modal_image_transform = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.cross_modal_image_transform.apply(utils.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(utils.init_weights)

        # Bert encoder
        bert_config_encoder = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_smiles_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
            is_decoder=False,
        )

        # cross-attention-encoder
        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(bert_config_encoder) for _ in range(config['num_top_layer'])])
        self.cross_modal_image_layers.apply(utils.init_weights)
        self.cross_modal_smiles_layers = nn.ModuleList([BertCrossLayer(bert_config_encoder) for _ in range(config['num_top_layer'])])
        self.cross_modal_smiles_layers.apply(utils.init_weights)

        # header predict layer
        self.cross_modal_image_pooler = predictor.Pooler(config["hidden_size"])
        self.cross_modal_image_pooler.apply(utils.init_weights)
        self.cross_modal_smiles_pooler = predictor.Pooler(config["hidden_size"])
        self.cross_modal_smiles_pooler.apply(utils.init_weights)

        self.mlm_probability = config['mlm_probability']
        self.mask_ratio = config['mask_ratio']

        # MLM Loss
        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = predictor.MLMHead(bert_config_encoder)
            self.mlm_score.apply(utils.init_weights)
        # itm Loss
        if config["loss_names"]["itm"] > 0 or config["loss_names"]["ocsr"] > 0 or config["loss_names"]["ocsr_finturn"] > 0:
            self.itm_score = predictor.ITMHead(config["hidden_size"]*2)
            self.itm_score.apply(utils.init_weights)
        # fpp Loss
        if config["loss_names"]["fpp"] > 0:
            self.fpp_score_smiles100 = predictor.FPPHead(config["hidden_size"], 100)
            self.fpp_score_smiles100.apply(utils.init_weights)
            self.fpp_score_images100 = predictor.FPPHead(config["hidden_size"], 100)
            self.fpp_score_images100.apply(utils.init_weights)

            self.fpp_score_smiles1000 = predictor.FPPHead(config["hidden_size"], 1000)
            self.fpp_score_smiles1000.apply(utils.init_weights)
            self.fpp_score_images1000 = predictor.FPPHead(config["hidden_size"], 1000)
            self.fpp_score_images1000.apply(utils.init_weights)

            self.fpp_score_smiles10000 = predictor.FPPHead(config["hidden_size"], 10000)
            self.fpp_score_smiles10000.apply(utils.init_weights)
            self.fpp_score_images10000 = predictor.FPPHead(config["hidden_size"], 10000)
            self.fpp_score_images10000.apply(utils.init_weights)

        # ===================== Downstream task ===================== #

        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"] and not self.hparams.config['is_pretrain']
        ):
            print('===================== load checkpoint for downstream task =====================')
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        if config["loss_names"]["MoleculeNet_classify"] > 0:
            self.MoleculeNet_classify_score = predictor.MoleculeClassify(config["hidden_size"]*2,drop_rate=config['drop_rate'])
            self.MoleculeNet_classify_score.apply(utils.init_weights)

        self.current_tasks = list()
        # train/loss
        metrics_utils.set_metrics(self)
        metrics_utils.set_task(self)

        self.focal_loss = FocalLoss(ignore_index=-100)

        # ===================== test_only ======================
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            print('===================== load checkpoint for test only =====================')
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            msg=self.load_state_dict(state_dict, strict=False)
            print(msg)
        if self.hparams.config["load_path"] == "" and ~self.hparams.config["test_only"]:
            print('===================== fineturning with nonPretrained =====================')

    def infer(
            self,images,smiles,images_false=None,smiles_false=None,
            image_token_type_idx = 1,
            mask_image = False,
            mask_smiles = False,
            change_smi = False,
    ):
        device = images.device
        smiles = list(smiles)  # tuple->list

        smiles = self.tokenizer(smiles, padding='max_length', truncation=True, max_length=202,
                                return_tensors="pt").to(device)
        if mask_smiles == True:
            input_ids = smiles.input_ids.clone()
            mlm_labels = input_ids.clone()
            probability_matrix = torch.full(mlm_labels.shape, self.mlm_probability)
            input_ids, mlm_labels = self.mask(input_ids, self.smiles_encoder.config.vocab_size, images.device,
                                                   targets=mlm_labels,
                                                   probability_matrix=probability_matrix)
            smiles.input_ids = input_ids
        else:
            mlm_labels = None

        smiles_embedding = self.smiles_encoder(smiles.input_ids, attention_mask=smiles.attention_mask, return_dict=True)
        smiles_embedding = self.cross_modal_smiles_transform(smiles_embedding.logits)
        smiles_masks = torch.ones((smiles_embedding.size(0), smiles_embedding.size(1)), dtype=torch.long, device=device)
        extend_smiles_masks = self.smiles_encoder.get_extended_attention_mask(smiles_masks, smiles_masks.size(), device)

        if mask_image == False:
            image_embedding, mask, ids_restore = self.visual_encoder.forward_encoder(images, mask_ratio=0)
        else:
            image_embedding, mask, ids_restore = self.visual_encoder.forward_encoder(images, mask_ratio=self.mask_ratio)

        image_masks = torch.ones((image_embedding.size(0), image_embedding.size(1)), dtype=torch.long,
                                 device=device)
        image_embedding = self.cross_modal_image_transform(image_embedding)

        extend_image_masks = self.smiles_encoder.get_extended_attention_mask(image_masks, image_masks.size(), device=device)

        smiles_embeds, image_embeds = (
            smiles_embedding + self.token_type_embeddings(torch.zeros_like(smiles_masks)),
            image_embedding
            + self.token_type_embeddings(
                torch.full_like(image_masks.long(), image_token_type_idx)
            ),
        )

        x, y = smiles_embeds, image_embeds
        for smiles_layer, image_layer in zip(self.cross_modal_smiles_layers, self.cross_modal_image_layers):
            x1 = smiles_layer(x, y, extend_smiles_masks, extend_image_masks)
            y1 = image_layer(y, x, extend_image_masks, extend_smiles_masks)
            x, y = x1[0], y1[0]

        smiles_feats = F.normalize(x, dim=-1)
        image_feats = F.normalize(y, dim=-1)

        if mask_image:
            pred = self.visual_encoder.forward_decoder(image_feats, ids_restore)
            patchify_img = self.visual_encoder.patchify(images)
        else:
            pred = None
            patchify_img = None

        # pooling
        cls_feats_smiles = self.cross_modal_smiles_pooler(smiles_feats)
        cls_feats_image = self.cross_modal_image_pooler(image_feats)
        cls_feats = torch.cat([cls_feats_image, cls_feats_smiles], dim=-1)

        ret = {
            "smiles_feats": smiles_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "cls_feats_smiles": cls_feats_smiles,
            "cls_feats_image": cls_feats_image,
            'smiles_masks': smiles.attention_mask,
            "image_masks": image_masks,
            "ids_restore": ids_restore,
            "mask": mask,
            "pred":pred,
            "imgs": patchify_img,
            "mlm_labels": mlm_labels,
        }
        return ret

    def forward(self, batch,testing = False):

        ret = dict()
        if len(self.current_tasks) == 0:
            images, smiles = batch
            ret.update(self.infer(images, smiles))
            return ret
    ####============================== Pretrain task ==================================####
        ## =========================== Image-Smiles Matching =========================== ##
        # Image-Smiles Matching
        if "itm" in self.current_tasks:
            ret.update(logger_itm(self,batch))

        ## =========================== Mask Language Modeling =========================== ##
        if "mlm" in self.current_tasks:
            ret.update(logger_mlm(self,batch))

        ## =========================== fingerprint predict =========================== ##
        if 'fpp' in self.current_tasks:
            ret.update(logger_fpp(self,batch))

    ####============================== downstream task ==================================####

        ## =========================== MoleculeNet classification =========================== ##
        if 'MoleculeNet_classify' in self.current_tasks:
            ret.update(compute_MoleculeNet_classify(self,batch,testing))

        return ret

    def training_step(self, batch, batch_idx):
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        return total_loss.float()

    def training_epoch_end(self, outs):
        metrics_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        metrics_utils.set_task(self)
        output = self(batch)
        return sum([v for k, v in output.items() if "loss" in k])

    def validation_epoch_end(self, outs):
        metrics_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        metrics_utils.set_task(self)
        output = self(batch,testing=True)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        return total_loss

    def test_epoch_end(self, outs):
        if self.config['is_pretrain'] == False:
            metrics_utils.test_epoch_auroc(self)
        else:   # 预训练
            metrics_utils.epoch_wrapup(self)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        return utils.set_schedule(self)

    def get_pretrained_tokenizer(self,tokenizer):
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                # 主线程
                AutoTokenizer.from_pretrained(tokenizer)
            torch.distributed.barrier()

        return AutoTokenizer.from_pretrained(tokenizer)

    def init_tokenizer(self):
        tokenizer = self.get_pretrained_tokenizer(self.config['tokenizer'])
        return tokenizer

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            # probability_matrix (bs,max_len) 0.15
            # (bs,max_len) bool
            masked_indices = torch.bernoulli(probability_matrix).bool()
        # pad_token_id=0,cls_token_id=12,mask_token_id=14
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

def myclip_pretrain(_config):
    model = myCLIP(_config)
    return model