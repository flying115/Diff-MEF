import functools
import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numbers
from einops import rearrange
import numpy as np
from Diffusion.Image_encoder import *
from Diffusion.Seg_encoder import SegEncoder
import functools
import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numbers
from einops import rearrange
import numpy as np
from Diffusion.Image_encoder import *
from Diffusion.Seg_encoder import SegEncoder

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class FeatureWiseAffine(nn.Module):
    def __init__(self, out_channels):
        super(FeatureWiseAffine, self).__init__()
        self.t_mlp = nn.Sequential(
            Swish(),
            #nn.Linear(128, out_channels )
            nn.Linear(192, out_channels )
        )

    def forward(self, x, t_emb):
        batch = x.shape[0]
        noise_feature = self.t_mlp(t_emb).view(batch, -1, 1, 1)
        x = x + noise_feature
        return x

class TextModulation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TextModulation, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(512,in_channels * 2),
            nn.SiLU(),
            nn.Linear(in_channels * 2, out_channels * 2)
        )
    def forward(self, x,text_emb):
        batch = x.shape[0]
        scale, shift = self.MLP(text_emb).view(batch, -1, 1, 1).chunk(2, dim=1)
        x = x * (1 + scale) + shift
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias,padding_mode):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias,padding_mode=padding_mode)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


        
class Attention(nn.Module):
    def __init__(self, dim, num_heads, padding_mode='reflect', bias=False) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.pwconv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, bias=bias, padding_mode=padding_mode, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x):
        _, _, h, w = x.shape

        qkv = self.dwconv(self.pwconv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class FeatureFusionModule(nn.Module):
    """ Layer attention module"""
    def __init__(self, in_dim, bias=True):
        super(FeatureFusionModule, self).__init__()
        self.chanel_in = in_dim
        self.temp = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d(self.chanel_in, self.chanel_in * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in * 3, self.chanel_in * 3, kernel_size=3, stride=1, padding=1,
                                    groups=self.chanel_in * 3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)

    def forward(self, x):
        B, N, C, H, W = x.size()
        x_input = x.view(B, N * C, H, W)

        qkv = self.qkv_dwconv(self.qkv(x_input))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(B, N, -1)
        k = k.view(B, N, -1)
        v = v.view(B, N, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temp
        attn = attn.softmax(dim=-1)
        out_1 = (attn @ v)
        out_1 = out_1.view(B, -1, H, W)

        out_1 = self.project_out(out_1)
        out_1 = out_1.view(B, N, C, H, W)
        out = out_1 + x
        out = out.view(B, -1, H, W)
        return out


##########################################################################
## Transformer banch

class TransBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, padding_mode, bias):
        super(TransBlock, self).__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.attn = Attention(dim, num_heads, padding_mode, bias)
        self.norm2 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias,padding_mode)
        self.affine = FeatureWiseAffine(dim)
        
    def forward(self, x, t_emb):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        out = self.affine(x,t_emb)
        return out



class RefineNet(nn.Module):
    def __init__(self,in_channel,T,dim,num_blocks,heads,ffn_expansion_factor,padding_mode,bias):

        super(RefineNet, self).__init__()
        tdim = dim * 4
        self.emb = TimeEmbedding(T, dim, tdim)
        
        self.patch_embed = OverlapPatchEmbed(in_channel, dim)
        
        self.refinement1 = TransBlock(dim=int(dim), num_heads=heads[0],ffn_expansion_factor=ffn_expansion_factor,
                                             padding_mode=padding_mode, bias=bias)
        
        self.refinement2 = TransBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                             padding_mode=padding_mode, bias=bias)
        
        self.refinement3 = TransBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                             padding_mode=padding_mode, bias=bias)
                                             
        self.fusion1 = FeatureFusionModule(in_dim=int(dim * 3))
        self.reduce_chan_in = nn.Conv2d(int(dim * 3), int(dim), kernel_size=1, bias=bias)
        
        
        layers = []
        for i in range(num_blocks[0]):
            trans_block = TransBlock(dim=dim,num_heads=heads[0],ffn_expansion_factor=ffn_expansion_factor,padding_mode=padding_mode, bias=bias)
            text_modulation = TextModulation(int(dim * 2),dim)
            layers.append(trans_block)
            layers.append(text_modulation)
                        
        self.encoder_level1 = nn.Sequential(*layers)
        self.down1_2 = Downsample(dim)

        layers = []
        for i in range(num_blocks[1]):
            trans_block = TransBlock(dim=int(dim * 2 ** 1),num_heads=heads[1],ffn_expansion_factor=ffn_expansion_factor,padding_mode=padding_mode, bias=bias)
            text_modulation = TextModulation(int(dim * 2 ** 2),int(dim * 2 ** 1))
            layers.append(trans_block)
            layers.append(text_modulation)
                
        self.encoder_level2 = nn.Sequential(*layers)

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3

        layers = []
        for i in range(num_blocks[2]):
            trans_block = TransBlock(dim=int(dim * 2 ** 2),num_heads=heads[2],ffn_expansion_factor=ffn_expansion_factor,padding_mode=padding_mode, bias=bias)
            text_modulation = TextModulation(int(dim * 2 ** 3),int(dim * 2 ** 2))
            layers.append(trans_block)
            layers.append(text_modulation)
                 
        self.encoder_level3 = nn.Sequential(*layers)
        
        self.down3_4 = Downsample(int(dim * 2 ** 2)) 
        

        layers = []
        for i in range(num_blocks[2]):
            trans_block = TransBlock(dim=int(dim * 2 ** 2),num_heads=heads[2],ffn_expansion_factor=ffn_expansion_factor,padding_mode=padding_mode, bias=bias)
            text_modulation = TextModulation(int(dim * 2 ** 3),int(dim * 2 ** 2))
            layers.append(trans_block)
            layers.append(text_modulation)
           
        self.decoder_level3 = nn.Sequential(*layers)

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)

        layers = []
        for i in range(num_blocks[1]):
            trans_block = TransBlock(dim=int(dim * 2 ** 1),num_heads=heads[1],ffn_expansion_factor=ffn_expansion_factor,padding_mode=padding_mode, bias=bias)
            text_modulation = TextModulation(int(dim * 2 ** 2),int(dim * 2 ** 1))
            layers.append(trans_block)
            layers.append(text_modulation)
            
        self.decoder_level2 = nn.Sequential(*layers)

        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.reduce_chan_level1 = nn.Conv2d(int(dim * 2 ** 1), int(dim), kernel_size=1, bias=bias)

        layers = []
        for i in range(num_blocks[0]):
            trans_block = TransBlock(dim=dim,num_heads=heads[0],ffn_expansion_factor=ffn_expansion_factor,padding_mode=padding_mode, bias=bias)
            text_modulation = TextModulation(int(dim * 2),dim)
            layers.append(trans_block)
            layers.append(text_modulation)
                        
        self.decoder_level1 = nn.Sequential(*layers)
        
        self.refinement4 = TransBlock(dim=int(dim), num_heads=heads[0],ffn_expansion_factor=ffn_expansion_factor,
                                             padding_mode=padding_mode, bias=bias)
        
        self.refinement5 = TransBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                             padding_mode=padding_mode, bias=bias)
        
        self.refinement6 = TransBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                             padding_mode=padding_mode, bias=bias)
                                             
        self.fusion2 = FeatureFusionModule(in_dim=int(dim * 3))
        self.reduce_chan_out = nn.Conv2d(int(dim * 3), int(dim), kernel_size=1, bias=bias)

        self.output = nn.Conv2d(int(dim), 3, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, inp_denoise, inp_concate, t, text_emb):
        
        device = t.device
        t = self.emb(t)
        B, C, H, W = inp_denoise.shape
        
        input = torch.cat((inp_denoise, inp_concate),dim=1)
        
        inp = self.patch_embed(input)
        
        refine1 = self.refinement1(inp,t)
        refine2 = self.refinement2(refine1,t)
        refine3 = self.refinement3(refine2,t)
        
        inp_fusion = torch.cat([refine1.unsqueeze(1),refine2.unsqueeze(1),refine3.unsqueeze(1)],dim=1)
        inp_fusion = self.fusion1(inp_fusion)
        inp_enc = self.reduce_chan_in(inp_fusion)
        
        inp_enc_level1 = inp_enc
        for layer in self.encoder_level1:
            if isinstance(layer, TransBlock):
                inp_enc_level1 = layer(inp_enc_level1, t)
            elif isinstance(layer, TextModulation):
                inp_enc_level1 = layer(inp_enc_level1, text_emb)
                
        out_enc_level1 = inp_enc_level1

        inp_enc_level2 = self.down1_2(out_enc_level1)
        for layer in self.encoder_level2:
            if isinstance(layer, TransBlock):
                inp_enc_level2 = layer(inp_enc_level2, t)
            elif isinstance(layer, TextModulation):
                inp_enc_level2 = layer(inp_enc_level2, text_emb)
            
        out_enc_level2 = inp_enc_level2

        inp_enc_level3 = self.down2_3(out_enc_level2)
        for layer in self.encoder_level3:
            if isinstance(layer, TransBlock):
                inp_enc_level3 = layer(inp_enc_level3, t)
            elif isinstance(layer, TextModulation):
                inp_enc_level3 = layer(inp_enc_level3, text_emb)
            
        out_enc_level3 = inp_enc_level3
      
        inp_dec_level3 = out_enc_level3
        for layer in self.decoder_level3:
            if isinstance(layer, TransBlock):
                inp_dec_level3 = layer(inp_dec_level3, t)
            elif isinstance(layer, TextModulation):
                inp_dec_level3 = layer(inp_dec_level3, text_emb)
                
        out_dec_level3 = inp_dec_level3

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        for layer in self.decoder_level2:
            if isinstance(layer, TransBlock):
                inp_dec_level2 = layer(inp_dec_level2, t)
            elif isinstance(layer, TextModulation):
                inp_dec_level2 = layer(inp_dec_level2, text_emb)
            
        out_dec_level2 = inp_dec_level2

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        
        for layer in self.decoder_level1:
            if isinstance(layer, TransBlock):
                inp_dec_level1 = layer(inp_dec_level1, t)
            elif isinstance(layer, TextModulation):
                inp_dec_level1 = layer(inp_dec_level1, text_emb)
        
        out_dec_level1 = inp_dec_level1

        refine4 = self.refinement4(out_dec_level1, t)
        refine5 = self.refinement5(refine4, t)
        refine6 = self.refinement6(refine5, t)
        out_fusion = torch.cat([refine4.unsqueeze(1),refine5.unsqueeze(1),refine6.unsqueeze(1)],dim=1)
        out_fusion = self.fusion2(out_fusion)
        out_fusion = self.reduce_chan_out(out_fusion)
        output = self.output(out_fusion) 
        
        return output
