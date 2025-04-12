from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from .bridge_former_layer_setting import Video_Bridge_Former
from .mcq_loss import sim_matrix, NormSoftmaxLoss


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device1 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mcq' in args.loss_names:
            self.bridge_layer = args.bridge_layer
            self.gate_way = args.gate_way
            ftr_dim = 512
            bridge_heads = ftr_dim // 64
            mcq_former = Video_Bridge_Former(layer_set=self.bridge_layer, embed_dim=ftr_dim, num_heads=bridge_heads,
                                             mlp_ratio=args.mlp_ratio, drop_path_rate=args.drop_path_rate)
            mcq_former.head = nn.Identity()
            mcq_former.pre_logits = nn.Identity()

            self.bridge_encoder = mcq_former
            self.bridge_encoder.fc = nn.Identity()
            self.bridge_proj = nn.Linear(ftr_dim, self.embed_dim)

            self.mcq_temperature = args.mcq_temperature
            self.mcq_loss_fn = NormSoftmaxLoss(temperature=self.mcq_temperature)


        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_question(self, question):
        x = self.base_model.encode_question(question)
        return x  # [13, bs, 77, 512]

    # mcq的compute text
    def compute_mcq_text(self, answer_data, question_data):
        answer_cls_embeddings = self.encode_text(answer_data)

        question_embeddings = self.encode_question(question_data)

        return answer_cls_embeddings, question_embeddings

    def compute_mcq_answer(self, image_hidden_embeddings, question_embeddings, question_mask, img_cls_emb):
        bridge_cls_embeddings = self.bridge_encoder(image_hidden_embeddings, question_embeddings, question_mask, img_cls_emb, self.gate_way)  # [bt, 768]

        bridge_cls_embeddings = self.bridge_proj(bridge_cls_embeddings)

        return bridge_cls_embeddings

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']

        image_feats, norm_image_feats, image_hidden_embeddings  = self.base_model.encode_image(images, return_hidden=True)

        text_feats = self.base_model.encode_text(caption_ids)

        i_feats = image_feats[:, 0, :].float()
        norm_i_feats = norm_image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'id' in self.current_task:
            # image_logits = self.classifier(i_feats.half()).float()
            # text_logits = self.classifier(t_feats.half()).float()
            image_logits = self.classifier(i_feats)
            text_logits = self.classifier(t_feats)
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        if 'mcq' in self.current_task:
            answer_data = batch['answer_ids']
            question_data = batch['question_ids']
            question_mask = batch['question_att_mask']

            answer_cls_embeddings, question_embeddings = \
                self.compute_mcq_text(answer_data, question_data)  # [bt, 512] [13, bt, 77, 512]

            answer_cls_embeddings = answer_cls_embeddings.to(self.device1)
            question_embeddings = question_embeddings.to(self.device1)
            image_hidden_embeddings = image_hidden_embeddings.to(self.device1)
            question_mask = question_mask.to(self.device1)


            img_cls_emb = i_feats.clone().to(self.device1)  # [bt, 512]
            bridge_cls_embeddings = \
                self.compute_mcq_answer(image_hidden_embeddings.float(), question_embeddings.float(), question_mask,
                                        img_cls_emb)

            sim_output = sim_matrix(answer_cls_embeddings, bridge_cls_embeddings)
            mcq_loss = self.mcq_loss_fn(sim_output)
            mcq_loss = mcq_loss.to(self.device0)
            ret.update({'mcq_loss': mcq_loss*self.args.mcq_loss_weight})


        return ret




def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    # convert_weights(model)
    return model

 # elif self.gate_way == "after_proj_gate_filter_v": ### proj后，单独图像, linear proj + gate sigmoid
#     img_cls_emb = i_feats.clone().to(self.device1)  # [bt, 512]
#     bridge_cls_embeddings = \
#         self.compute_mcq_answer(image_hidden_embeddings.float(), question_embeddings.float(), question_mask, img_cls_emb)
# elif self.gate_way == "before_proj_v_t": ### proj后，图像文本
#     img_cls_emb = image_hidden_embeddings[-1,:,0,:]  # 最后一层的cls编码
#     txt_cls_emb = transformer_last_cls[:,0,:].to(self.device1) # 最后一层的cls编码, layernorm之前的 transformer_last_cls[bt,77, 512]
#     img_cls_emb_expand = img_cls_emb.unsqueeze(0).unsqueeze(2)
#     txt_cls_emb_expand = txt_cls_emb.unsqueeze(0).unsqueeze(2)
#     img_filtered_emb = img_cls_emb_expand * image_hidden_embeddings
#     question_filtered_emb = txt_cls_emb_expand * question_embeddings
#     bridge_cls_embeddings = \
#         self.compute_mcq_answer(img_filtered_emb.float(), question_filtered_emb.float(), question_mask)

# bridge_cls_embeddings = \
#     self.compute_mcq_answer(image_hidden_embeddings.float(), question_embeddings.float(), question_mask)
