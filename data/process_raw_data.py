import pandas as pd
import numpy as np
from scipy import signal
import os
import re
import lmdb
import pickle
# 新增绘图库
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # 无界面后端，适配服务器运行


# 1. 工具函数（极简）
def butter_filter(data, fs, cutoff, btype='band'):
    b, a = signal.butter(2, np.array(cutoff)/(0.5*fs), btype=btype)
    return signal.filtfilt(b, a, data, axis=0)

def normalize_length(data, target_len):
    return data[:target_len] if len(data) > target_len else np.pad(data, (0, target_len-len(data)), 'constant')


# 2. LMDB构建函数（精简，无修改）
def build_lmdb(input_dir, lmdb_path, label_path):
    try:
        label_map = {int(r['num']):[r['happy'],r['angry'],r['sad'],r['neutral']] 
                     for _,r in pd.read_csv(label_path).iterrows()}
        print(f"✅ 加载{len(label_map)}个标签")
    except Exception as e:
        print(f"❌ 标签读取失败：{type(e).__name__}|{e}")
        return

    env = lmdb.open(lmdb_path, map_size=1024*1024*1024)
    all_keys = []
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    with env.begin(write=True) as txn:
        sorted_files = sorted(
            [f for f in os.listdir(input_dir) if f.endswith('.csv')],
            key=lambda x: int(re.search(r'(\d+)', x).group(1))  # 提取文件名中数字作为排序键
        )
        for f in sorted_files:
            try:
                simple_num = int(re.search(r'(\d+)', f).group(1))  # 直接提取文件名数字
                if simple_num not in label_map:
                    print(f"⚠️ 跳过{f}：simple_num={simple_num}无标签")
                    continue

                df = pd.read_csv(os.path.join(input_dir,f), skiprows=2, on_bad_lines='skip')
                eeg = df[['EEG.FP1','EEG.FP2']].replace(['', ' '], np.nan).apply(pd.to_numeric, errors='coerce').dropna(how='all').values
                fnirs = df[[c for c in df.columns if c.startswith('FNIRS.S')]].replace(['', ' '], np.nan).apply(pd.to_numeric, errors='coerce').dropna(how='all').values
                
                eeg = butter_filter(eeg, 250, [1,30], 'band')
                fnirs = butter_filter(fnirs, 25, [0.1], 'low')

                eeg = normalize_length(eeg.flatten(), 5000).reshape(2,10,250).astype(np.float32)
                fnirs = normalize_length(fnirs.flatten(), 4000).reshape(16,10,25).astype(np.float32)
                key = str(simple_num)
                txn.put(key.encode(), pickle.dumps({
                        'eeg':eeg, 
                        'fnirs':fnirs, 
                        'label':np.array(label_map[simple_num], np.int32)
                    }))
                all_keys.append(key)
                print(f"✅ {f}→Key{key}")
            except Exception as e:
                print(f"❌ {f}失败：{type(e).__name__}|{e}")

        if all_keys:
            txn.put(b'__keys__', pickle.dumps({
                'train':all_keys[:60], 
                'val':all_keys[60:80], 
                'test':all_keys[80:80]
            }))

    env.close()
    print(f"\n🎉 写入{len(all_keys)}个样本")


# 3. LMDB测试函数（新增绘图功能）
def test_lmdb(lmdb_path, save_plot_dir=None):
    # 初始化图片保存路径（默认在LMDB目录下创建plot文件夹）
    if save_plot_dir is None:
        save_plot_dir = os.path.join(lmdb_path, 'signal_plots')
    os.makedirs(save_plot_dir, exist_ok=True)
    print(f"📸 图片将保存到：{save_plot_dir}")

    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            # 测试1：读数据集划分
            keys_dict = pickle.loads(txn.get(b'__keys__'))
            print("="*50)
            print("1. 数据集划分：")
            for mode, keys in keys_dict.items():
                print(f"   {mode}: {len(keys)}个样本（keys：{keys[:3]}...）")

            # 测试2：读样本+验证+绘图
            sample_keys = [k for k in keys_dict['train'] + keys_dict['val'] if k]
            if not sample_keys:
                print("⚠️ 无有效样本，终止测试")
                return

            print("\n2. 样本数据验证+信号绘图（前2个样本）：")
            for key in sample_keys[:2]:
                data = pickle.loads(txn.get(key.encode()))
                print(f"\n   Key{key}：")
                # 修复：打印形状而非整个数组
                print(f"   - eeg形状：{data['eeg'].shape}（目标：(2,10,250)）")
                print(f"   - fnirs形状：{data['fnirs'].shape}（目标：(16,10,25)）")
                print(f"   - 标签：{data['label']}（目标：4维）")

                # -------------------------- 新增：三维转二维（Channel, 时间段*时间点） --------------------------
                # EEG：(2,10,250) → (2, 2500)（2通道，每通道2500个时间点）
                eeg_2d = data['eeg'].reshape(data['eeg'].shape[0], -1)  # 合并后两维
                # fNIRS：(16,10,25) → (16, 250)（16通道，每通道250个时间点）
                fnirs_2d = data['fnirs'].reshape(data['fnirs'].shape[0], -1)
                print(f"   - eeg二维形状：{eeg_2d.shape}（目标：(2,2500)）")
                print(f"   - fnirs二维形状：{fnirs_2d.shape}（目标：(16,250)）")

                # -------------------------- 新增：绘制信号图 --------------------------
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))  # 2行1列子图
                fig.suptitle(f'Signal Plot - Key{key} (Label: {data["label"]})', fontsize=14, fontweight='bold')

                # 子图1：EEG信号（2通道）
                eeg_fs = 250  # EEG采样率
                eeg_time = np.arange(eeg_2d.shape[1]) / eeg_fs  # 时间轴（单位：秒）
                ax1.plot(eeg_time, eeg_2d[0], label='EEG.FP1', color='#1f77b4', linewidth=0.8)
                ax1.plot(eeg_time, eeg_2d[1], label='EEG.FP2', color='#ff7f0e', linewidth=0.8)
                ax1.set_title('EEG Signals (2 Channels)', fontsize=12)
                ax1.set_xlabel('Time (s)', fontsize=10)
                ax1.set_ylabel('Amplitude', fontsize=10)
                ax1.legend()
                ax1.grid(alpha=0.3)

                # 子图2：fNIRS信号（16通道，用colormap区分）
                fnirs_fs = 25  # fNIRS采样率
                fnirs_time = np.arange(fnirs_2d.shape[1]) / fnirs_fs  # 时间轴
                cmap = plt.get_cmap('tab20')  # 多通道配色
                for i in range(fnirs_2d.shape[0]):
                    ax2.plot(fnirs_time, fnirs_2d[i] + i*0.5,  # 通道间偏移，避免重叠
                             label=f'FNIRS.S{i+1}', color=cmap(i), linewidth=0.8)
                ax2.set_title('fNIRS Signals (16 Channels)', fontsize=12)
                ax2.set_xlabel('Time (s)', fontsize=10)
                ax2.set_ylabel('Amplitude (Offset)', fontsize=10)
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)  # 图例放右侧
                ax2.grid(alpha=0.3)

                # 调整布局+保存图片
                plt.tight_layout()
                plot_path = os.path.join(save_plot_dir, f'key_{key}_signals.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   📸 信号图保存：{plot_path}")

        env.close()
        print("="*50)
        print("✅ LMDB测试+绘图完成！")
    except Exception as e:
        print(f"❌ LMDB测试失败：{type(e).__name__}|{e}")


# 4. 主入口
if __name__ == "__main__":
    # 配置路径
    INPUT_DIR = "/dev/yowant2_cfs/tianyi/csp/RN_Project/data/raw_data/EEG_Fnirs/S_1"
    LABEL_PATH = "/dev/yowant2_cfs/tianyi/csp/RN_Project/data/raw_data/EEG_Fnirs/S_1/label/label.csv"
    LMDB_PATH = "/dev/yowant2_cfs/tianyi/csp/RN_Project/data/clean_data/S_1"

    # 执行构建+测试（含绘图）
    build_lmdb(INPUT_DIR, LMDB_PATH, LABEL_PATH)
    # test_lmdb(LMDB_PATH)  # 默认保存图片到 LMDB_PATH/signal_plots