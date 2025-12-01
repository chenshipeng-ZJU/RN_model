import torch

def fnirs(fnirs_data: torch.Tensor, age: int = 30):
    """
    从fNIRS原始数据计算HbO和HbR浓度变化
    输入形状: (B, C_total, S, T) → B=批次, C_total=总通道数(偶数), S=时间段, T=每个时间段样本数
    输出形状: (B, C, S, T) → C=C_total//2（物理通道数）
    """
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
    dpf_735, dpf_850 = calc_dpf(age, 735), calc_dpf(age, 850)

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
    combined_data = torch.cat([oxy_data_reshaped, dxy_data_reshaped], dim=1)
    
    return combined_data


def eeg(eeg_data: torch.Tensor):
    """
    从EEG数据提取时频特征（基于STFT）
    输入形状: (B, C, S, T) → B=批次, C=通道数, S=时间段, T=每个时间段样本数
    输出形状: total_feat=(B, C, P, 5) → P=时间窗口数, 5个频段；low/high_calib=(B, C, P, 1)
    """
    win_length, hop_length, sfreq = 100, 50, 250  # STFT参数
    if eeg_data.ndim == 3:
        eeg_data = eeg_data.unsqueeze(2)  # 补全维度为(B, C, 1, T)
    b, c, s, t = eeg_data.shape

    # 1. STFT计算功率谱
    eeg_flat = eeg_data.reshape(b * c, s * t)  # 展平为(B*C, S*T)
    stft = torch.stft(
        eeg_flat, n_fft=win_length, win_length=win_length, hop_length=hop_length,
        window=torch.hann_window(win_length, device=eeg_data.device),
        return_complex=True, center=False
    )  # STFT形状: (B*C, freq_bins, P)
    power = torch.abs(stft).reshape(b, c, stft.shape[1], stft.shape[2])**2  # 功率谱: (B, C, freq_bins, P)

    # 2. 提取5个频段特征（delta-theta-alpha-beta-gamma）
    freq_axis = torch.fft.rfftfreq(win_length, d=1/sfreq, device=eeg_data.device)
    freq_bands = [(1,4), (4,8), (8,13), (13,30), (30,45)]  # 5个频段范围
    all_feats = []
    for fmin, fmax in freq_bands:
        # 筛选频段内的频率点，计算均值作为频段特征
        freq_mask = (freq_axis >= fmin) & (freq_axis <= fmax)
        band_feat = torch.mean(power[:, :, freq_mask, :], dim=2).unsqueeze(-1)  # (B, C, P, 1)
        all_feats.append(band_feat)

    # 3. 拼接频段特征，计算高低频校准特征
    total_feat = torch.cat(all_feats, dim=-1)  # (B, C, P, 5)
    low_calib = torch.mean(total_feat[..., :2], dim=-1, keepdim=True)  # 低频（delta+theta）校准
    high_calib = torch.mean(total_feat[..., 2:], dim=-1, keepdim=True)  # 高频（alpha+beta+gamma）校准

    return total_feat, low_calib, high_calib


# -------------------------- 测试代码 --------------------------
if __name__ == "__main__":
    # FNIRS测试
    fnirs_tensor = torch.rand(2, 16, 10, 25)  # 输入: (B=2, C_total=16, S=10, T=25)
    combined_fnirs = fnirs(fnirs_tensor)
    print("FNIRS结果形状:")
    print(combined_fnirs.shape)  # 预期: (2, 8, 10, 25)

    # EEG测试
    eeg_tensor = torch.rand(2, 2, 10, 250)  # 输入: (B=2, C=2, S=10, T=250)
    total_feat, low_calib, high_calib = eeg(eeg_tensor)
    print("\nEEG结果形状:")
    print(f"总特征: {total_feat.shape}, 低频校准: {low_calib.shape}, 高频校准: {high_calib.shape}")  # 预期: (2,2,P,5), (2,2,P,1), (2,2,P,1)