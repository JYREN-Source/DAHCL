import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        assert data_format in ["channels_last", "channels_first"]
        self.data_format = data_format

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, (self.weight.shape[0],), self.weight, self.bias, self.eps)
        else:  # channels_first
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            return self.weight[:, None] * x + self.bias[:, None]


class GRN(nn.Module):
    """Global Response Normalization"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ==============================
#   FRFE Block
# ==============================
class FRFEBlock(nn.Module):
    def __init__(self, dim, drop_path=0., dropout_prob=0.2):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        x = self.dropout(x)
        return residual + self.drop_path(x)


class FE(nn.Module):
    def __init__(self, in_chans=1, depths=[1, 1, 3, 1], dims=[40, 80, 160, 320], drop_path_rate=0.):
        super().__init__()
        self.num_stages = len(depths)

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        ))

        for i in range(self.num_stages - 1):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            ))

        self.stages = nn.ModuleList()
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0

        for i in range(self.num_stages):
            stage = nn.Sequential(*[
                FRFEBlock(dim=dims[i], drop_path=dp_rates[cur + j])
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(self.num_stages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        return x.permute(0, 2, 1)


class Classifier(nn.Module):

    def __init__(self, input_dim, output_dim, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU()
        )
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x, return_embed=False):

        x = x.mean(dim=-1)  # (B, C) = (B, input_dim)
        embed = self.proj(x)  # (B, embed_dim)
        logits = self.fc(embed)  # (B, num_classes)

        if return_embed:
            return logits, embed
        return logits

class DomainModulator(nn.Module):
    def __init__(self, num_classes, embed_dim, hidden_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.cls_mlp = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

        self.feat_mlp = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, d_vec):
        if d_vec.dim() == 2:
            d = d_vec.mean(dim=0)
        elif d_vec.dim() == 1:
            d = d_vec
        else:
            raise ValueError

        v_cls = self.cls_mlp(d)  # [K]
        v_feat = self.feat_mlp(d)  # [D]

        gamma_raw = torch.outer(v_cls, v_feat)  # [K, D]
        return gamma_raw


class Model(nn.Module):
    def __init__(self,
                 num_classes=10,
                 in_chans=1,
                 num_domains=2,
                 embed_dim=64,
                 use_domain_bias=True):

        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.feature_extractor = FE(in_chans=in_chans)

        self.classifier = Classifier(input_dim=320, output_dim=num_classes, embed_dim=embed_dim)

        self.domain_classifier = Classifier(input_dim=320, output_dim=num_domains, embed_dim=embed_dim)

        self.domain_modulator = DomainModulator(
            num_classes=num_classes,
            embed_dim=embed_dim,
            hidden_dim=128
        )

    def forward(self, x, alpha=None):

        features = self.feature_extractor(x)  # (B, C=320, L)

        logits = self.classifier(features)  # -> (B, num_classes)

        domain_logits = self.domain_classifier(features)  # -> (B, num_domains)

        return logits, domain_logits