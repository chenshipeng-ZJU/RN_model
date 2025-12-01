import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGFeatureExtractor:
    """EEG特征提取器：通过STFT提取多频段特征"""
    def __init__(self, sfreq=250, win_length=100, hop_length=50, freq_bands=None):
        self.sfreq = sfreq
        self.win_length = win_length
        self.hop_length = hop_length
        self.freq_bands = freq_bands or [(1,4), (4,8), (8,13), (13,30), (30,45)]  # 5个频段范围

    def __call__(self, eeg_data: torch.Tensor):
        b, c, s, t = eeg_data.shape
        # 1. STFT计算功率谱
        eeg_flat = eeg_data.reshape(b * c, s * t)  # 展平为(B*C, S*T)
        stft = torch.stft(
            eeg_flat, n_fft=self.win_length, win_length=self.win_length, hop_length=self.hop_length,
            window=torch.hann_window(self.win_length, device=eeg_data.device),
            return_complex=True, center=False
        )  # STFT形状: (B*C, freq_bins, P)
        power = torch.abs(stft).reshape(b, c, stft.shape[1], stft.shape[2])**2  # 功率谱: (B, C, freq_bins, P) freq_bins = win_length/2 + 1

        # 2. 提取5个频段特征（delta-theta-alpha-beta-gamma）
        freq_axis = torch.fft.rfftfreq(self.win_length, d=1/self.sfreq, device=eeg_data.device)
        all_feats = []
        for fmin, fmax in self.freq_bands:
            # 筛选频段内的频率点，计算均值作为频段特征
            freq_mask = (freq_axis >= fmin) & (freq_axis <= fmax)
            band_feat = torch.mean(power[:, :, freq_mask, :], dim=2).unsqueeze(-1)  # (B, C, P, 1)
            all_feats.append(band_feat)

        # 3. 拼接频段特征，计算高低频校准特征
        total_feat = torch.cat(all_feats, dim=-1)  # (B, C, P, 5)
        low_feat = torch.mean(total_feat[..., :2], dim=-1, keepdim=True)  # 低频（delta+theta）校准
        high_feat = torch.mean(total_feat[..., 2:], dim=-1, keepdim=True)  # 高频（alpha+beta+gamma）校准

        return power.permute(0, 1, 3, 2), low_feat, high_feat       # 调整为(B, C, P, freq_bins) 

class FNIRSFeatureExtractor:
    """fNIRS特征提取器：通过滑动窗口提取统计特征"""
    def __init__(self, age=30):
        self.age = age
    
    def __call__(self, fnirs_data: torch.Tensor):
        b, c_total, s, t = fnirs_data.shape
        c = c_total // 2  # 物理通道数（每个通道含2个波长）

        # 数据展平：(B*C_total, S*T)，便于批量计算
        raw_data = fnirs_data.reshape(b * c_total, s * t)
        num_samples = s * t
        baseline_window = 100  # 基线窗口长度（样本数）

        # 1. 计算基线和光密度(OD)
        baseline = torch.mean(raw_data[:, :baseline_window], dim=1, keepdim=True)  # (B*C_total, 1)
        od_data = -torch.log10(raw_data / (baseline + 1e-10))  # 避免log(0)，OD形状：(B*C_total, S*T)

        # 2. 预计算固定参数（DPF、距离、消光系数）
        # 差分路径因子(DPF)
        def calc_dpf(age_val, lambda_val):
            return 223.3 + 0.05624 * age_val**0.8493 - 5.723e-7 * lambda_val**3 + 0.001245 * lambda_val**2 - 0.9025 * lambda_val
        dpf_735, dpf_850 = calc_dpf(self.age, 735), calc_dpf(self.age, 850)

        # 光源-探测器距离（cm），每个物理通道对应1个距离
        distance = torch.tensor([3.2, 2.5298, 2.5298, 1.0, 3.2, 2.5298, 2.5298, 1.0], device=fnirs_data.device, dtype=fnirs_data.dtype)

        # 血红蛋白消光系数（/mM/cm，转换为自然对数）
        oxy_735, deoxy_735 = torch.tensor(0.1225 / 2.303, device=fnirs_data.device, dtype=fnirs_data.dtype), torch.tensor(0.2961 / 2.303, device=fnirs_data.device, dtype=fnirs_data.dtype)
        oxy_850, deoxy_850 = torch.tensor(0.2526 / 2.303, device=fnirs_data.device, dtype=fnirs_data.dtype), torch.tensor(0.1798 / 2.303, device=fnirs_data.device, dtype=fnirs_data.dtype)

        # 3. 初始化结果张量
        oxy_data = torch.zeros((b * c, num_samples), device=fnirs_data.device, dtype=fnirs_data.dtype)
        dxy_data = torch.zeros((b * c, num_samples), device=fnirs_data.device, dtype=fnirs_data.dtype)

        # 4. 批量计算HbO和HbR（修正的比尔-朗伯定律）
        denom = oxy_735 * deoxy_850 - oxy_850 * deoxy_735  # 公共分母，避免重复计算
        if abs(denom) < 1e-10:
            raise ValueError("计算分母接近零，无法反演浓度")

        for i in range(b * c):
            # 提取当前物理通道的两个波长数据
            ch_idx1, ch_idx2 = 2 * i, 2 * i + 1
            od_735, od_850 = od_data[ch_idx1], od_data[ch_idx2]
            L = distance[i % c]  # 匹配当前物理通道的距离

            # 浓度反演公式
            oxy_data[i] = (1 / L) * (od_735 * deoxy_850 / dpf_735 - od_850 * deoxy_735 / dpf_850) / denom
            dxy_data[i] = (1 / L) * (od_735 * oxy_850 / dpf_735 - od_850 * oxy_735 / dpf_850) / denom

        # 5. 重塑结果并在通道维度拼接
        oxy_data_reshaped = oxy_data.reshape(b, c, s, t)
        dxy_data_reshaped = dxy_data.reshape(b, c, s, t)
        fnirs_feat = torch.cat([oxy_data_reshaped, dxy_data_reshaped], dim=1)
        
        return fnirs_feat
    
class PatchEmbedding(nn.Module):
    """模态特征嵌入层：将原始特征映射到高维空间，添加位置编码"""
    def __init__(self, eeg_sfreq=250, d_model=256):
        super().__init__()
        self.d_model = d_model
        self.proj_time = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.SiLU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.SiLU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.SiLU(),
        )
        self.proj_time_final = nn.Sequential(
            nn.Linear(eeg_sfreq, d_model*2),
            nn.SiLU(),
            nn.Linear(d_model*2, d_model)
        )
        self.proj_freq = nn.Sequential(
            nn.Linear(eeg_sfreq//2+1, d_model*2),
            nn.SiLU(),
            nn.Linear(d_model*2, d_model)
        )
        self.pos_encoding = nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model)  # 深度可分离卷积位置编码
        self.mask_encoding = nn.Parameter(torch.zeros(eeg_sfreq), requires_grad=False)

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, eeg_sfreq = x.shape
        # 缺失值填充
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding
        
        # 时域特征投影
        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, eeg_sfreq)
        time_emb = self.proj_time(mask_x)
        time_emb = time_emb.permute(0, 2, 1, 3).contiguous().view(bz*ch_num*patch_num, eeg_sfreq)
        time_emb = self.proj_time_final(time_emb)
        time_emb = time_emb.contiguous().view(bz, ch_num, patch_num, self.d_model)

        # 频域特征投影
        mask_x = mask_x.contiguous().view(bz*ch_num*patch_num, eeg_sfreq)
        freq_emb = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        freq_emb = torch.abs(freq_emb).contiguous().view(bz*ch_num*patch_num, eeg_sfreq//2+1)
        freq_emb = self.proj_freq(freq_emb)
        freq_emb = freq_emb.contiguous().view(bz, ch_num, patch_num, self.d_model)

        #位置特征投影
        patch_emb = time_emb + freq_emb
        positional_embedding = self.pos_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding
        return patch_emb


class TransformerEncoderLayer(nn.Module):
    """Transformer层：包含模态内注意力和跨模态注意力"""
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.num_heads = nhead // 2  # 单模态头数（总头数需为偶数）

        # 注意力模块
        self.eeg_attn_s = nn.MultiheadAttention(d_model//2, self.num_heads, dropout=dropout, batch_first=True)  # EEG空间注意力
        self.eeg_attn_t = nn.MultiheadAttention(d_model//2, self.num_heads, dropout=dropout, batch_first=True)  # EEG时间注意力
        self.fnirs_attn_s = nn.MultiheadAttention(d_model//2, self.num_heads, dropout=dropout, batch_first=True)  # fNIRS空间注意力
        self.fnirs_attn_t = nn.MultiheadAttention(d_model//2, self.num_heads, dropout=dropout, batch_first=True)  # fNIRS时间注意力
        # 跨模态注意力块
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)  # 跨模态注意力

        # ====================================专家组门控设计===========================================
        # 创新设计：专家组模型讨论的机制
        self.eeg_gate = nn.Linear(1, d_model//2)
        # self.fnirs_gate = nn.Linear(1, d_model//2)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), 
            nn.SiLU(), 
            nn.Dropout(dropout), 
            nn.Linear(dim_feedforward, d_model)
            )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, eeg_feat, fnirs_feat, eeg_calib):
        # 模态内注意力计算
        eeg_attn = self._modal_attn(eeg_feat, is_eeg=True, eeg_calib=eeg_calib)
        fnirs_attn = self._modal_attn(fnirs_feat, is_eeg=False)

        # 跨模态注意力
        batch_eeg, C_eeg, P_eeg, d_eeg = eeg_attn.shape
        batch_fnirs, C_fnirs, P_fnirs, d_fnirs = fnirs_attn.shape
        eeg_flat = eeg_attn.flatten(1, 2)  # (B, C_eeg*P_eeg, d_model)
        fnirs_flat = fnirs_attn.flatten(1, 2)  # (B, C_fnirs*P_fnirs, d_model)

        # 跨注意力计算（EEG为查询Q从fNIRS获取空间信息，fNIRS为键值K,V）
        cross_out_eeg = self.cross_attn(eeg_flat, fnirs_flat, fnirs_flat)[0]
        cross_out_eeg = cross_out_eeg.unflatten(1, (C_eeg, P_eeg))  # 恢复EEG形状
        # （fNIRS为查询Q从EEG获取时间信息，EEG为键值K,V）
        cross_out_fnirs = self.cross_attn(fnirs_flat, eeg_flat, eeg_flat)[0]
        cross_out_fnirs = cross_out_fnirs.unflatten(1, (C_fnirs, P_fnirs))  # 恢复fNIRS形状
        # 残差融合+归一化（双向融合）
        eeg_out = self.norm(eeg_feat + self.dropout(eeg_attn + cross_out_eeg))
        fnirs_out = self.norm(fnirs_feat + self.dropout(fnirs_attn + cross_out_fnirs))  # fNIRS融合跨模态信息

        # 前馈网络
        eeg_out = eeg_out + self.dropout(self.ffn(eeg_out))
        fnirs_out = fnirs_out + self.dropout(self.ffn(fnirs_out))

        return eeg_out, fnirs_out

    def _modal_attn(self, x, is_eeg, eeg_calib=None):
        """模态内注意力：空间（通道间）+时间（patch间）"""
        batch_size, ch_num, patch_num, d_model = x.shape
        half_d_model = d_model // 2
        # 特征拆分
        xs, xt = x[..., :half_d_model], x[..., half_d_model:]
        # 校准门控
        if is_eeg:
            gate = self.eeg_gate 
            # EEG 的低频成分具有更稳定、更广泛的空间分布模式，适合作为 “空间显著性” 的指标。EEG 的高频成分对事件的时间锁定性更强，能够精确反映神经活动的瞬时变化，适合作为 “时间显著性” 的指标。
            gate_s, gate_t = torch.sigmoid(gate(eeg_calib[0])), torch.sigmoid(gate(eeg_calib[1]))
            # 校准门维度匹配
            if gate_s.shape[2] != xs.shape[2]:
                gate_s = gate_s.repeat_interleave(xs.shape[2]//gate_s.shape[2]+1, dim=2)[:, :, :xs.shape[2], :]
            if gate_t.shape[2] != xt.shape[2]:
                gate_t = gate_t.repeat_interleave(xt.shape[2]//gate_t.shape[2]+1, dim=2)[:, :, :xt.shape[2], :]
            xs, xt = xs * gate_s[..., :half_d_model], xt * gate_t[..., :half_d_model]

        # 1. 空间注意力（通道间）
        xs_in = xs.transpose(1, 2).reshape(-1, ch_num, half_d_model)
        attn_s, _ = (self.eeg_attn_s if is_eeg else self.fnirs_attn_s)(xs_in, xs_in, xs_in)
        attn_s = attn_s.reshape(batch_size, patch_num, ch_num, half_d_model).transpose(1, 2) + xs  # 残差

        # 2. 时间注意力（patch间）
        xt_in = xt.reshape(-1, patch_num, half_d_model) 
        attn_t, _ = (self.eeg_attn_t if is_eeg else self.fnirs_attn_t)(xt_in, xt_in, xt_in)
        attn_t = attn_t.reshape(batch_size, ch_num, patch_num, half_d_model) + xt  # 残差

        return self.dropout(torch.cat([attn_s, attn_t], dim=-1))

class TransformerEncoder(nn.Module):
    """多模态Transformer编码器：堆叠多个Transformer层"""
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, eeg_feat, fnirs_feat, eeg_calib):
        out_eeg, out_fnirs = eeg_feat, fnirs_feat

        for layer in self.layers:
            out_eeg, out_fnirs = layer(out_eeg, out_fnirs, eeg_calib)

        return torch.cat([out_eeg, out_fnirs], dim=1) # 融合两模态特征


class RN_Model(nn.Module):
    """多模态脑信号融合模型：EEG+fNIRS特征融合，输出奖励信号特征"""
    def __init__(self, eeg_sfreq=250, d_model=256, nhead=8, num_layers=1, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        # 特征嵌入
        self.eeg_embed = PatchEmbedding(eeg_sfreq, d_model)   
        self.fnirs_embed = PatchEmbedding(eeg_sfreq, d_model)       
        
        # Transformer融合
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        
        # 预训练的输出头
        self.proj_out = nn.Sequential(
                    nn.Linear(d_model, d_model*2),
                    nn.SiLU(),
                    nn.Linear(d_model*2, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, eeg_sfreq),
        )        
        # 权重初始化
        self.apply(self._weights_init)

    def _weights_init(self, m):
        """权重初始化：兼容低PyTorch版本，适配GELU/SiLU"""
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(
                m.weight, 
                mode='fan_out', 
                nonlinearity='leaky_relu', 
                a=0.01  # 小斜率模拟GELU/SiLU的平滑特性
            )
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, eeg, fnirs, eeg_mask=None, fnirs_mask=None, gate_feat=None):
        # 特征嵌入
        eeg_embed = self.eeg_embed(eeg, eeg_mask)
        fnirs_embed  = self.fnirs_embed(fnirs, fnirs_mask)
        # print("特征嵌入结束...")
        # print(eeg_embed.shape, fnirs_embed.shape)

        # Transformer融合
        fused_feat = self.encoder(
            eeg_embed, fnirs_embed,
            eeg_calib=(gate_feat[0], gate_feat[1])
        )
        # print("特征融合结束...")
        # print(fused_feat.shape)

        # 预训练输出
        fused_feat = fused_feat.reshape(-1, fused_feat.shape[3])     # 展平形状：(batch_size*(ch_eeg + ch_fnirs)*patch, d_model)
        out = self.proj_out(fused_feat)                              # 形状：(batch_size*(ch_eeg + ch_fnirs)*patch, eeg_sfreq)
        # print("预训练输出结束...")
        # print(out.shape)

        return out

def generate_mask(bz, ch_num, patch_num, device, mask_ratio=0.2):
    mask = torch.zeros((bz, ch_num, patch_num), dtype=torch.long, device=device)
    mask = mask.bernoulli_(mask_ratio)
    return mask

if __name__ == '__main__':
    """测试模型前向传播"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    channels_eeg = 2   # EEG通道总数
    channels_fnirs = 16  # Fnirs通道总数
    patch_num = 10       # 时间段数
    trials_eeg = 250    # 时间点数
    trials_fnirs = 25   # 时间点数
    num_of_classes = 4  # 分类回归数

    # 测试数据（B, C, P, T）
    eeg = torch.randn(batch_size, channels_eeg, patch_num, trials_eeg).to(device)         # EEG: 4样本, 2通道, 10空间点, 250时间点
    fnirs = torch.randn(batch_size, channels_fnirs, patch_num, trials_fnirs).to(device)   # fNIRS: 4样本, 16通道, 10空间点, 25时间点
    
    # 特征提取
    # 2.特征提取器
    eeg_extractor = EEGFeatureExtractor()
    fnirs_extractor = FNIRSFeatureExtractor()
    eeg_feat, eeg_low, eeg_high = eeg_extractor(eeg)
    fnirs_feat = fnirs_extractor(fnirs)
    # print("特征提取结束...")
    # print(eeg_feat.shape, eeg_low.shape, eeg_high.shape)
    # print(fnirs_feat.shape)
    
    # 输入数据处理维度对齐
    fnirs_3D= fnirs.reshape(batch_size*channels_fnirs, patch_num, trials_fnirs)
    fnirs = torch.nn.functional.interpolate(
        input=fnirs_3D,                 # 输入张量
        size=trials_eeg,              # 目标尺寸 (时间点, 特征数)。注意，这里要指定所有空间维度的目标大小
        mode='linear',               # 插值模式，'linear'表示线性插值
        align_corners=False          # 控制插值对齐方式，在深度学习中通常设为False以获得更好的效果
    )
    fnirs = fnirs.reshape(batch_size, channels_fnirs, patch_num, trials_eeg)
    # print("维度对齐结束...")
    # print(eeg.shape, fnirs.shape) 

    # 20%缺失率掩码
    eeg_mask = generate_mask(batch_size, channels_eeg, patch_num, mask_ratio=0.2, device=device)
    fnirs_mask = generate_mask(batch_size, channels_fnirs, patch_num, mask_ratio=0.2, device=device)

    # 模型测试
    model = RN_Model(num_layers=1).to(device)
    out = model(eeg, fnirs, eeg_mask, fnirs_mask, gate_feat=(eeg_low, eeg_high))

    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    while True:
        pass  # 保持运行以检查显存使用情况
