from torch import nn, einsum
import torch
from einops import rearrange, repeat
from functools import partial
from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def attn(q, k, v):

    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out

def attn_mask(q, k, v, mask):

    sim = einsum('b i d, b j d -> b i j', q, k)
    mask = (1.0 - mask) * -10000.0
    mask = repeat(mask, 'b d -> b r d', r=q.shape[1])
    sim = sim + mask
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VarAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 initialize='random'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask, whether, einops_from, einops_to, **einops_dims):
        h = self.num_heads
        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        q *= self.scale
        mask = repeat(mask, 'b d -> (b r) d', r=self.num_heads)
        n_f = int(einops_dims['f'])
        # splice out CLS token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))
        # let CLS token attend to key / values of all patches across time and space
        if whether is not True:
            cls_out = attn(cls_q, k, v)
        else:
            cls_mask = mask[:, 0:1]
            mask_ = mask[:, 1:]
            mask_ = mask_.repeat(1, n_f)
            mask_tile = torch.cat((cls_mask, mask_), dim=1)
            cls_out = attn_mask(cls_q, k, v, mask_tile)
        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r=r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)

        # attention
        if whether is not True:
            out = attn(q_, k_, v_)
        else:
            mask_tile = mask.repeat_interleave(n_f, 0)
            out = attn_mask(q_, k_, v_, mask_tile)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim=1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        # to out
        x = self.proj(out)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 initialize='random'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q_question = nn.Linear(512, dim * 1, bias=qkv_bias)
        #self.kv_video = nn.Linear(dim, dim * 1, bias=qkv_bias)
        # print("---cross-attention kv dim 512 to 768------")
        self.kv_video = nn.Linear(768, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, question, mask, whether, einops_from, einops_to, **einops_dims):
        h = self.num_heads
        # project x to q, k, v vaalues
        q = self.q_question(question)
        k, v = self.kv_video(x).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        q *= self.scale
        cls_q = q[:, 0:1]
        q_ = q[:, 1:]

        cls_out = attn(cls_q, k, v)

        n_f = int(einops_dims['f'])
        q_ = q_.repeat_interleave(n_f, 0)
        k, v = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (k, v))

        out = attn(q_, k, v)
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim=1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        # to out
        x = self.proj(out)
        x = self.proj_drop(x)
        return x

class task_guided_gate(nn.Module):
    def __init__(self, in_dim=512, out_dim=768):
        super().__init__()
        self.gate_linear_proj = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(in_dim)


    def forward(self, img_cls_emb, hidden_embeds, gate_way):
        # img_cls_emb [bt, 512]
        # hidden_embeds [bt, 193, 768]
        if "res_filter" in gate_way:
            img_cls_emb = self.gate_linear_proj(self.ln(img_cls_emb))
            img_cls_emb_gate = torch.sigmoid(img_cls_emb)  # [batch, 768]
            img_cls_emb_gate_expand = img_cls_emb_gate.unsqueeze(1)
            img_cls_emb_expand = img_cls_emb.unsqueeze(1)
            img_filtered_emb = hidden_embeds * img_cls_emb_gate_expand + img_cls_emb_expand  # [13,bt,193,768]
        elif "noproj_filter" in gate_way:
            img_cls_emb = torch.sigmoid(img_cls_emb)  # [batch, 768]
            img_cls_emb_expand = img_cls_emb.unsqueeze(1)
            img_filtered_emb = hidden_embeds * img_cls_emb_expand  # [13,bt,193,768]
        elif "dot_filter" in gate_way:
            img_cls_emb = torch.sigmoid(self.gate_linear_proj(self.ln(img_cls_emb)))  # [batch, 768]
            img_cls_emb_expand = img_cls_emb.unsqueeze(1)
            img_filtered_emb = hidden_embeds * img_cls_emb_expand  # [13,bt,193,768]

        return img_filtered_emb





class Video_Bridge_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = VarAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.crossattn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.bridgeattn = VarAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.bridge_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm4 = norm_layer(768)
        # self.norm5 = norm_layer(dim) # 文本
        self.norm5 = norm_layer(512) # 文本
        self.norm6 = norm_layer(dim)
        self.norm7 = norm_layer(dim)

        # 增加gate
        self.gate_filter = task_guided_gate(in_dim=512, out_dim=768)


    def forward(self, x_bridge, x, question, mask, layer, einops_from_space, einops_to_space,
                einops_from_time, einops_to_time, time_n, space_f, img_cls_emb, gate_way):
        x = self.gate_filter(img_cls_emb, x, gate_way)
        cross_out = self.crossattn(self.norm4(x[:, 1:]), self.norm5(question), mask, False,
                                   einops_from_space, einops_to_space, f=space_f)

        if layer == 0:
            bridge_temp = cross_out
        else:
            bridge_temp = cross_out + x_bridge

        space_bridge_output = self.bridgeattn(self.norm7(bridge_temp), mask, True, einops_from_space,
                                    einops_to_space, f=space_f)
        space_bridge_residual = bridge_temp + self.drop_path(space_bridge_output)
        x_bridge_after = space_bridge_residual + self.drop_path(self.bridge_mlp(self.norm6(space_bridge_residual))) # bridge的一层transformer结构得到bridge的特征

        return x_bridge_after

class Video_Bridge_Former(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `Space-Time Transformer` from Frozen-in-time  - by Max Bain.
        https://arxiv.org/abs/2104.00650

    """

    def __init__(self, layer_set, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 num_frames=1):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim，即mlp的中间隐层维度扩大多少倍
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
            num_frames: (int) maximum number of frames expected as input， num_frames根据CC3M数据集的设置为1
        """
        super().__init__()


        idx_s = [6,7,8,9,10,11]


        self.layer_set = "hou6"
        self.idx_s = idx_s
        depth = len(self.idx_s)

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Video_Bridge_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)

        # Representation layer
        if representation_size: # self.pre_logits = nn.Identity()
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()  # nn.Identity()

        # trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.cls_token, std=.02)

        # if num_frames > 1, then we perform ViT inflation and initialise time attention to zero so not necessary.
        if num_frames == 1:
            self.apply(self._init_weights)

        # einops transformations
        self.einops_from_space = 'b (f n) d'
        self.einops_to_space = '(b f) n d'
        self.einops_from_time = 'b (f n) d'
        self.einops_to_time = '(b n) f d'


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, visual_hiddens, question, mask, img_cls_emb, gate_way):

        f = 1

        layer = 0
        first_layer_id = self.idx_s[layer]
        x_bridge = visual_hiddens[first_layer_id]
        for blk in self.blocks: # len(self.blocks)=12
            if self.layer_set == "ini":
                id_ = int(layer / 2)
            else:
                id_ = self.idx_s[layer]
            question_temp = question[id_]
            x = visual_hiddens[id_]

            x_bridge = blk(x_bridge, x, question_temp, mask, layer, self.einops_from_space,
                              self.einops_to_space, self.einops_from_time, self.einops_to_time, time_n=0,
                              space_f=f, img_cls_emb=img_cls_emb, gate_way=gate_way)
            layer = layer + 1

        x_bridge = self.norm2(x_bridge)[:, 0]
        x_bridge = self.pre_logits(x_bridge)

        return x_bridge


    def forward(self, x, question, mask, img_cls_emb, gate_way):
        x_bridge = self.forward_features(x, question, mask, img_cls_emb, gate_way)
        x_bridge = self.head(x_bridge)
        return x_bridge