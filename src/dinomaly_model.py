"""Anomalib Dinomaly modules with rank-8 encoder adapters and offline reload.

Reuses Apache-2.0 Anomalib 2.2.0's Dinomaly implementation and DINOv2 loader.
See docs/research_sources.md for upstream references and integration differences.
"""

from functools import partial
from pathlib import Path

import torch
from torch import nn

from anomalib.models.components import GaussianBlur2d
from anomalib.models.image.dinomaly.torch_model import DinomalyModel, DecoderViTBlock
from anomalib.models.image.dinomaly.components import CosineHardMiningLoss, DinomalyMLP, LinearAttention
from anomalib.models.image.dinomaly.components.dinov2_loader import DinoV2Loader


class StableLinearAttention(LinearAttention):
    """Upstream's positive-kernel attention with FP32 accumulation and epsilon.

    Dinomaly's author documents the epsilon fix for training on new datasets.
    FP32 accumulation also prevents half-precision overflow of the token sums.
    """
    def forward(self, x):
        batch, length, dimension = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, dimension // self.num_heads).permute(2, 0, 3, 1, 4)
        with torch.autocast(device_type=x.device.type, enabled=False):
            q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
            q, k = torch.nn.functional.elu(q) + 1, torch.nn.functional.elu(k) + 1
            kv = k.transpose(-2, -1) @ v
            denominator = (q * k.sum(dim=-2, keepdim=True)).sum(-1, keepdim=True).clamp_min(1e-6)
            output = (q @ kv) / denominator
        output = output.transpose(1, 2).reshape(batch, length, dimension).to(x.dtype)
        return self.proj_drop(self.proj(output)), kv


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 8):
        super().__init__()
        self.base = linear
        self.base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, linear.in_features))
        self.b = nn.Parameter(torch.zeros(linear.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)
        self.scale = 1.0

    def forward(self, x):
        return self.base(x) + torch.nn.functional.linear(torch.nn.functional.linear(x, self.a), self.b) * self.scale


class SurgicalDinomaly(DinomalyModel):
    def __init__(self, pretrained: bool = True):
        # Assemble the upstream modules directly so a saved model reload performs
        # no pretrained-network download. The forward/loss/map code is upstream.
        nn.Module.__init__(self)
        loader = DinoV2Loader(Path(__file__).resolve().parents[1] / "models/pretrained")
        self.encoder = loader._create_model("dinov2_reg", "small", 14)
        if pretrained:
            loader._load_weights(self.encoder, "dinov2_reg", "small", 14)
        self.encoder.requires_grad_(False)
        for block in self.encoder.blocks[:10]:
            block.attn.qkv = LoRALinear(block.attn.qkv)
            block.attn.proj = LoRALinear(block.attn.proj)
        self.bottleneck = nn.ModuleList([DinomalyMLP(in_features=384, hidden_features=1536,
            out_features=384, act_layer=nn.GELU, drop=0.2, bias=False, apply_input_dropout=True)])
        self.decoder = nn.ModuleList([DecoderViTBlock(dim=384, num_heads=6, mlp_ratio=4.0,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0.0, attn=StableLinearAttention)
            for _ in range(8)])
        self.target_layers = list(range(2, 10))
        self.fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.remove_class_token = False
        self.gaussian_blur = GaussianBlur2d(sigma=4, channels=1, kernel_size=5)
        self.loss_fn = CosineHardMiningLoss()
        from anomalib.models.image.dinomaly.lightning_model import Dinomaly
        Dinomaly._initialize_trainable_modules(nn.ModuleList([self.bottleneck, self.decoder]))

    def get_encoder_decoder_outputs(self, images):
        # Upstream's encoder is under no_grad. Keep its weights frozen while
        # allowing gradients through the inserted LoRA matrices.
        tokens = self.encoder.prepare_tokens(images)
        encoded = []
        for index, block in enumerate(self.encoder.blocks[:10]):
            tokens = block(tokens)
            if index in self.target_layers:
                encoded.append(tokens)
        side = int((encoded[0].shape[1] - 1 - self.encoder.num_register_tokens) ** 0.5)
        reconstructed = self._fuse_feature(encoded)
        for block in self.bottleneck:
            reconstructed = block(reconstructed)
        decoded = []
        for block in self.decoder:
            reconstructed = block(reconstructed, attn_mask=None)
            decoded.append(reconstructed)
        decoded.reverse()
        en = [self._fuse_feature([encoded[i] for i in indices]) for indices in self.fuse_layer_encoder]
        de = [self._fuse_feature([decoded[i] for i in indices]) for indices in self.fuse_layer_decoder]
        return self._process_features_for_spatial_output(en, side), self._process_features_for_spatial_output(de, side)
