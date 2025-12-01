import torch
import torch.nn as nn
from base_model import RN_Model

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.backbone = RN_Model(num_layers=config.num_layers)

        if config.use_pretrained_weights:
            self.backbone.load_state_dict(torch.load(config.dir_pretrained_weights, map_location=config.device)["model_state_dict"])
            # 冻结backbone所有参数（不参与训练）
            if config.frozen:
                for param in self.backbone.parameters():
                    param.requires_grad = False
        self.backbone.proj_out = nn.Identity()
        
        # 微调的回归输出头
        self.fused_flatten_dim = (config.eeg_channels + config.fnirs_channels) * config.patch_num * config.d_model    # 融合后的通道数
        self.shared_feature_layers = nn.Sequential(
            nn.Linear(self.fused_flatten_dim, 512),
            nn.LayerNorm(512),        # 使用 LayerNorm 更适合Transformer的输出
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256), 
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(config.dropout)
        )
        self.exclusive_layers = nn.ModuleList([nn.Sequential(nn.Linear(256, 64), nn.SiLU()) for _ in range(config.num_of_classes)])
        self.output_heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(config.num_of_classes)])
        self.sigmoid = nn.Sigmoid()

    def forward(self, eeg, fnirs, gate_feat=None):
        fused_feat = self.backbone(eeg, fnirs, gate_feat=gate_feat)
        fused_feat = fused_feat.reshape(-1, self.fused_flatten_dim)   # 形状：(batch_size, (ch_eeg + ch_fnirs)*patch*d_model)

        scores = []
        shared_feat = self.shared_feature_layers(fused_feat)    # 提取出共享特征的形状：(batch_size, 256)
        for i in range(self.config.num_of_classes):
            exclusive_feat = self.exclusive_layers[i](shared_feat)  # (batch_size, 64)
            raw_score = self.output_heads[i](exclusive_feat)  # (batch_size, 1)
            normalized_score = self.sigmoid(raw_score)  # (batch_size, 1)
            scores.append(normalized_score)
        final_scores = torch.cat(scores, dim=1)  # 合并4个评分形状：(batch_size, 4)
        print(final_scores)
        return final_scores


