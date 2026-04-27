import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
import numbers


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


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
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)


    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)
    def forward(self, x):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)
    def forward(self, x):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x



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

class Attention(nn.Module):
    def __init__(self, dim, num_heads, padding_mode='reflect', bias=False) -> None:
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.pwconv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, bias=bias, padding_mode=padding_mode, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.feedforward = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=bias, groups=dim, padding_mode=padding_mode),
            nn.GELU()
        )

    def forward(self, x):
        attn = self.norm(x)
        _, _, h, w = attn.shape

        qkv = self.dwconv(self.pwconv(attn))
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
        out = self.feedforward(out + x)
        return out


class ResBlockWithAttn(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False, ratio=16):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = Attention(out_ch, num_heads=1, padding_mode='reflect', bias=False)
        else:
            self.attn = nn.Identity()
        self.initialize()
        
    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)
        h = h + self.shortcut(x)
        h = self.attn(h)
        return h

class UNet(nn.Module):
    def __init__(self, T, ch, channel_mults, attn, num_res_blocks, dropout):
        super().__init__()
        t_dim = ch * 4
        self.t_emb = TimeEmbedding(T, ch, t_dim)
        
        self.start_conv = nn.Conv2d(9, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  
        now_ch = ch
        for i, mult in enumerate(channel_mults):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlockWithAttn(
                    in_ch=now_ch, out_ch=out_ch, tdim=t_dim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(channel_mults) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlockWithAttn(now_ch, now_ch, t_dim, dropout, attn=True),
            ResBlockWithAttn(now_ch, now_ch, t_dim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlockWithAttn(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=t_dim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        self.initialize()
    def initialize(self):
        init.xavier_uniform_(self.start_conv.weight)
        init.zeros_(self.start_conv.bias)
        init.xavier_uniform_(self.final_conv[-1].weight, gain=1e-5)
        init.zeros_(self.final_conv[-1].bias)

    def forward(self, input, t):
        device = t.device
        t = self.t_emb(t)
        # Downsampling
        x = self.start_conv(input)
        feats = [x]
        for layer in self.downblocks:
            if isinstance(layer, ResBlockWithAttn):
                x = layer(x, t)
                feats.append(x)
            else:
                x = layer(x)
                feats.append(x)
        # Middle
        for layer in self.middleblocks:
            x = layer(x, t)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlockWithAttn):
                x = torch.cat([x, feats.pop()], dim=1)
                x = layer(x, t)
            else:
                x = layer(x)
        x = self.final_conv(x)

        assert len(feats) == 0
        return x
