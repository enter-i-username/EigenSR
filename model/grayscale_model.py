import numpy as np
import torch
from torch import nn
from .vit import VisionTransformer
from .lora import LoRA
from .conv import ResBlock, UpX2Res
import utils


class GrayscaleSR(nn.Module):

    def __init__(self,
                 scale,
                 hidden_channels,
                 need_lora=False,
                 lora_r=4):
        super().__init__()

        self.scale = scale

        self.head = HeadTail(
            in_channels=1,
            hidden_channels=hidden_channels,
            out_channels=64,
            scale=scale,
            upx2=False,
            norm='IN'
        )

        self.body = Transformer()

        # register lora for each attn layer if needed
        if need_lora:
            for _i, _layer in enumerate(self.body.body.encoder.layers):
                attn = _layer.self_attn
                lora = LoRA(attn.embed_dim, lora_r)
                attn.set_lora(lora)

            for _i, _layer in enumerate(self.body.body.decoder.layers):
                # self attn
                attn = _layer.self_attn
                lora = LoRA(attn.embed_dim, lora_r)
                attn.set_lora(lora)

                # mha
                attn = _layer.multihead_attn
                lora = LoRA(attn.embed_dim, lora_r)
                attn.set_lora(lora)

        self.tail = HeadTail(
            in_channels=64,
            hidden_channels=hidden_channels,
            out_channels=1,
            scale=scale,
            upx2=True,
            norm='IN'
        )

    def forward(self, x):
        # b, c, deg_h, deg_w = x.shape

        # ps = Transformer.DefaultViTParams.patch_size
        # x = utils.pad_if_smaller(x, size=(ps, ps))

        x = self.head(x)
        residual = x
        x = self.body(x)
        x = x + residual
        x = self.tail(x)

        # x = utils.crop_center_if_larger(x, size=(deg_h * self.scale, deg_w * self.scale), scale=self.scale)
        return x

    def load_frozen_body(self, ckpt):
        self.body.load_from_ckpt(ckpt)

        for name, param in self.body.named_parameters():
            if name.find('lora') == -1:
                param.requires_grad = False

        return self

    def load_body(self, ckpt):
        self.body.load_from_ckpt(ckpt)
        return self

    def save_trainable(self, ckpt):
        state_dict = {_n: _p for _n, _p in self.named_parameters() if _p.requires_grad}
        torch.save(state_dict, ckpt)
        return self

    def load_head_tail(self, ckpt):
        state_dict = torch.load(ckpt, map_location='cpu')
        ht_dict = {_k: _v for _k, _v in state_dict.items()
                     if _k.find('head') == 0 or _k.find('tail') == 0}
        assert len(ht_dict) >= 1

        model_state_dict = self.state_dict()
        model_state_dict.update(ht_dict)
        self.load_state_dict(model_state_dict, strict=True)
        return self

    def load_lora(self, ckpt):
        state_dict = torch.load(ckpt, map_location='cpu')
        lora_dict = {_k: _v for _k, _v in state_dict.items()
                     if _k.find('lora') >= 0}
        assert len(lora_dict) >= 1

        model_state_dict = self.state_dict()
        model_state_dict.update(lora_dict)
        self.load_state_dict(model_state_dict, strict=True)
        return self


class HeadTail(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 scale,
                 upx2=False,
                 norm='BN'):
        super().__init__()

        assert scale & (scale - 1) == 0
        num_hidden = int(np.log2(scale))

        self.block1 = ResBlock(
            in_channels,
            hidden_channels,
            stride=1,
            norm=norm,
            learnable_norm=True,
            need_out_relu=True
        )

        hidden_block = UpX2Res if upx2 else ResBlock
        hidden_params = {
            'in_channels': hidden_channels,
            'out_channels': hidden_channels,
            'stride': 1,
            'norm': norm,
            'learnable_norm': True,
            'need_out_relu': True
        }

        self.blocks = nn.Sequential(
            *[hidden_block(**hidden_params) for _ in range(num_hidden)]
        )

        self.blockn = ResBlock(
            hidden_channels,
            out_channels,
            stride=1,
            norm='None',
            learnable_norm=False,
            need_out_relu=False
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.blocks(x)
        x = self.blockn(x)
        return x


class Transformer(nn.Module):

    def __init__(self):
        super().__init__()

        self.body = VisionTransformer(
            img_dim=self.DefaultViTParams.patch_size,
            patch_dim=self.DefaultViTParams.patch_dim,
            num_channels=self.DefaultViTParams.n_feats,
            embedding_dim=self.DefaultViTParams.n_feats *
                          self.DefaultViTParams.patch_dim *
                          self.DefaultViTParams.patch_dim,
            num_heads=self.DefaultViTParams.num_heads,
            num_layers=self.DefaultViTParams.num_layers,
            hidden_dim=self.DefaultViTParams.n_feats *
                       self.DefaultViTParams.patch_dim *
                       self.DefaultViTParams.patch_dim *
                       4,
            num_queries=self.DefaultViTParams.num_queries,
            dropout_rate=self.DefaultViTParams.dropout_rate,
            mlp=self.DefaultViTParams.no_mlp,
            pos_every=self.DefaultViTParams.pos_every,
            no_pos=self.DefaultViTParams.no_pos,
            no_norm=self.DefaultViTParams.no_norm,
        )

    def forward(self, x):
        return self.body(x, 0)

    def load_from_ckpt(self, ckpt):
        state_dict = torch.load(ckpt, map_location='cpu')
        body_dict = {_k: _v for _k, _v in state_dict.items() if _k.find('body') == 0}
        assert len(body_dict) >= 1
        body_dict['body.query_embed.weight'] = body_dict['body.query_embed.weight'][0:1, ...]

        model_state_dict = self.state_dict()
        model_state_dict.update(body_dict)
        self.load_state_dict(model_state_dict, strict=True)

        return self

    class DefaultViTParams:
        n_feats = 64
        patch_size = 48
        patch_dim = 3
        num_heads = 12
        num_layers = 12
        num_queries = 1
        dropout_rate = 0
        no_mlp = False
        pos_every = False
        no_pos = False
        no_norm = False
