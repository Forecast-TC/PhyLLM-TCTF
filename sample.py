import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from torch.utils.data import Dataset
import torch
from collections import Counter
from utils import set_seed


class TyphoonDataConfig:
    """热带气旋数据集配置类"""

    def __init__(self, args):
        """
        Args:
            args: Parsed arguments from ``config.get_args()``.
        """
        self.data_root = args.data_root
        self.cma_path = os.path.join(args.data_root, 'CMA')
        self.era5_path = os.path.join(args.data_root, 'ERA5')
        self.himawari_path = os.path.join(args.data_root, 'Himawari')
        self.seq_len = args.seq_len
        self.pred_offset = args.pred_len
        self.train_ratio = args.train_ratio
        self.val_ratio = args.val_ratio

        self.himawari_channels = args.himawari_channels
        self.era5_variables_single = args.era5_single
        self.era5_variables_pres = args.era5_pres


class TyphoonDataLoader:

    def __init__(self, config: TyphoonDataConfig, seed: int = 42):
        self.config = config
        self._seed = seed
        self.samples = None
        self.train_samples = None
        self.val_samples = None
        self.test_samples = None
        self.typhoon_stats = {}
        self.typhoon_id_mapping = {}
        self.all_typhoon_ids = []

    def build_samples(self):
        self.samples = self._build_aligned_samples()
        typhoon_counter = 0

        seen_typhoons = set()

        for sample in self.samples:
            typhoon_key = f"{sample['year']}_{sample['typhoon_name']}"
            if typhoon_key not in seen_typhoons:
                self.typhoon_id_mapping[typhoon_counter] = typhoon_key
                self.all_typhoon_ids.append(typhoon_counter)
                seen_typhoons.add(typhoon_key)
                typhoon_counter += 1

            sample['typhoon_id'] = [tid for tid, key in self.typhoon_id_mapping.items() if key == typhoon_key][0]

        return self.samples

    def split_dataset(self, train_years, test_years):
        if self.samples is None:
            self.build_samples()

        year_dict = {}
        for sample in self.samples:
            year = sample['year']
            if year not in year_dict:
                year_dict[year] = []
            year_dict[year].append(sample)
        all_available_years = sorted(year_dict.keys(), key=lambda x: int(x))
        print(f"数据集中包含的所有年份: {all_available_years}")

        self.typhoon_stats['total_typhoons'] = len(self.typhoon_id_mapping)
        self.typhoon_stats['total_samples'] = len(self.samples)
        self.typhoon_stats['total_years'] = len(all_available_years)

        for year in train_years + test_years:
            if year not in year_dict:
                raise ValueError(f"指定的年份 {year} 不在数据集中，请检查！")

        overlap_train_test = set(train_years) & set(test_years)
        if overlap_train_test:
            raise ValueError(f"训练集和测试集年份重叠: {overlap_train_test}")

        self.typhoon_stats['split_method'] = 'manual'
        self.typhoon_stats['train_years'] = train_years
        self.typhoon_stats['test_years'] = test_years

        raw_train_samples = [sample for year in train_years for sample in year_dict.get(year, [])]
        self.test_samples = [sample for year in test_years for sample in year_dict.get(year, [])]

        set_seed(self._seed)

        train_typhoon_ids = list({sample['typhoon_id'] for sample in raw_train_samples})
        np.random.shuffle(train_typhoon_ids)
        val_typhoon_count = int(len(train_typhoon_ids) * self.config.val_ratio)
        val_typhoon_ids = train_typhoon_ids[:val_typhoon_count]
        train_typhoon_ids = train_typhoon_ids[val_typhoon_count:]

        self.train_samples = [sample for sample in raw_train_samples if sample['typhoon_id'] in train_typhoon_ids]
        self.val_samples = [sample for sample in raw_train_samples if sample['typhoon_id'] in val_typhoon_ids]

        self.typhoon_stats['train_typhoon_sample_counts'] = Counter(
            sample['typhoon_id'] for sample in self.train_samples
        )
        self.typhoon_stats['val_typhoon_sample_counts'] = Counter(
            sample['typhoon_id'] for sample in self.val_samples
        )
        self.typhoon_stats['test_typhoon_sample_counts'] = Counter(
            sample['typhoon_id'] for sample in self.test_samples
        )

        for set_name, samples in [('train', self.train_samples),
                                  ('val', self.val_samples),
                                  ('test', self.test_samples)]:
            typhoon_ids = set(sample['typhoon_id'] for sample in samples)
            typhoon_names = [self.typhoon_id_mapping[tid] for tid in typhoon_ids]

            self.typhoon_stats[f'{set_name}_typhoons'] = {
                'count': len(typhoon_ids),
                'ids': sorted(typhoon_ids),
                'names': typhoon_names,
                'years': train_years if set_name in ['train', 'val'] else test_years
            }

        self._print_dataset_stats()

        return self.train_samples, self.val_samples, self.test_samples

    def _print_dataset_stats(self):
        print(f"数据集中包含的台风条数：{self.typhoon_stats['total_typhoons']}")
        print(f"数据集中包含的总样本数：{self.typhoon_stats['total_samples']}")

        print(f"\n数据集划分方式: 手动指定年份（训练/验证从训练年份中按台风比例划分）")
        print(
            f"训练年份: {len(self.typhoon_stats['train_years'])} 年 ({self.typhoon_stats['train_years']})，总样本数: {len(self.train_samples) + len(self.val_samples)}")
        print(
            f"训练集: 占训练年份台风的{self.config.train_ratio * 100}%，样本数: {len(self.train_samples)}，台风数: {self.typhoon_stats['train_typhoons']['count']}")
        print(
            f"验证集: 占训练年份台风的{self.config.val_ratio * 100}%，样本数: {len(self.val_samples)}，台风数: {self.typhoon_stats['val_typhoons']['count']}")
        print(
            f"测试集: {len(self.typhoon_stats['test_years'])} 年 ({self.typhoon_stats['test_years']})，样本数: {len(self.test_samples)}，台风数: {self.typhoon_stats['test_typhoons']['count']}")
        print('-' * 120)

        print("训练集台风ID、名称及样本数:")
        train_counts = self.typhoon_stats['train_typhoon_sample_counts']  # 取出训练集样本数统计
        for tid in self.typhoon_stats['train_typhoons']['ids']:
            typhoon_name = self.typhoon_id_mapping[tid]
            sample_count = train_counts.get(tid, 0)  # 获取当前台风的样本数（默认0，避免异常）
            print(f"  ID: {tid:2d} -> 名称: {typhoon_name:15s} | 样本数: {sample_count:3d}")

        print("\n验证集台风ID、名称及样本数:")
        val_counts = self.typhoon_stats['val_typhoon_sample_counts']  # 取出验证集样本数统计
        for tid in self.typhoon_stats['val_typhoons']['ids']:
            typhoon_name = self.typhoon_id_mapping[tid]
            sample_count = val_counts.get(tid, 0)
            print(f"  ID: {tid:2d} -> 名称: {typhoon_name:15s} | 样本数: {sample_count:3d}")

        print("\n测试集台风ID、名称及样本数:")
        test_counts = self.typhoon_stats['test_typhoon_sample_counts']  # 取出测试集样本数统计
        for tid in self.typhoon_stats['test_typhoons']['ids']:
            typhoon_name = self.typhoon_id_mapping[tid]
            sample_count = test_counts.get(tid, 0)
            print(f"  ID: {tid:2d} -> 名称: {typhoon_name:15s} | 样本数: {sample_count:3d}")
        print('-' * 120)

    def get_typhoon_samples(self, typhoon_id, dataset_type='test'):
        if typhoon_id not in self.typhoon_id_mapping:
            raise ValueError(f"台风ID {typhoon_id} 不存在")

        dataset = {
            'train': self.train_samples,
            'val': self.val_samples,
            'test': self.test_samples
        }.get(dataset_type, self.test_samples)

        if not dataset:
            raise ValueError(f"{dataset_type}数据集尚未加载")

        return [sample for sample in dataset if sample['typhoon_id'] == typhoon_id]

    def _build_aligned_samples(self):
        samples = []
        cma_root = self.config.cma_path
        era5_root = self.config.era5_path
        himawari_root = self.config.himawari_path

        for year in os.listdir(cma_root):
            cma_year_dir = os.path.join(cma_root, year)
            if not os.path.isdir(cma_year_dir):
                continue

            for typhoon_file in os.listdir(cma_year_dir):
                if not typhoon_file.endswith('.csv'):
                    continue

                typhoon_name = typhoon_file.replace('.csv', '')
                file_path = os.path.join(cma_year_dir, typhoon_file)

                df = pd.read_csv(file_path, header=None)
                if df.shape[1] < 4:
                    continue

                df.columns = ['time', 'idensity', 'lat', 'lon', 'pressure', 'wind']
                df['timestamp'] = pd.to_datetime(df['time'].astype(str), format='%Y%m%d%H')
                df = df.sort_values(by='timestamp').reset_index(drop=True)

                for i in range(len(df) - self.config.seq_len - self.config.pred_offset + 1):
                    input_rows = df.iloc[i:i + self.config.seq_len]
                    target_rows = df.iloc[i + self.config.seq_len: i + self.config.seq_len + self.config.pred_offset]
                    valid = True
                    era5_paths, himawari_paths = [], []
                    for ts in input_rows['timestamp']:
                        ts_str = ts.strftime('%Y%m%d%H')
                        era5_path = os.path.join(era5_root, year, typhoon_name, f"{ts_str}.nc")
                        himawari_path = os.path.join(himawari_root, year, typhoon_name, f"{ts_str}.nc")
                        if not (os.path.exists(era5_path) and os.path.exists(himawari_path)):
                            valid = False
                            break
                        era5_paths.append(era5_path)
                        himawari_paths.append(himawari_path)

                    if not valid:
                        continue

                    sample = {
                        'typhoon_name': typhoon_name,
                        'year': year,
                        'track_input': input_rows[['lat', 'lon']].values,
                        'target_lats': target_rows['lat'].values,
                        'target_lons': target_rows['lon'].values,
                        # --------------------------------------------------------------------------------
                        'era5_paths': era5_paths,
                        'himawari_paths': himawari_paths,
                        'timestamps': [t.strftime('%Y%m%d%H') for t in input_rows['timestamp']]
                    }
                    samples.append(sample)

        dataset_name = os.path.basename(self.config.data_root)
        h_shape = dataset_name.split('_')[0]
        e_shape = dataset_name.split('_')[0]
        print(
            f'使用的数据集是：{self.config.data_root}\tHimawari的形状是：{h_shape}×{h_shape}\tERA5的形状是：{e_shape}×{e_shape}\n')
        print(f'使用的通道是：Himawari：{self.config.himawari_channels}\t'
              f'ERA5：{self.config.era5_variables_single} '
              f'{self.config.era5_variables_pres}\n')
        print(f"CMA路径: {self.config.cma_path}")
        print(f"ERA5路径： {self.config.era5_path}")
        print(f"Himawari路径： {self.config.himawari_path}")
        print(f"✅ 多步预测配置：输入序列长度={self.config.seq_len}，预测步数={self.config.pred_offset}\n")

        print(f"✅ 构建完成，共生成样本数: {len(samples)}")
        return samples


class TyphoonDataset(Dataset):
    def __init__(self, samples, config: TyphoonDataConfig):
        self.samples = samples
        self.config = config
        self._printed = True

        self.era5_norm_range = {
            # 单层变量（无压力层）
            'sst': (273, 312),
            'msl': (87000, 108400),
            'v10': (-55,  +55),
            'u10': (-55, +55),
            'z': {
                200: (108000, 126000),
                500: (48000, 59000),
                850: (12000, 15000),
                925: (6000, 8500)
            },
            'u': {
                200: (-120,  +120),
                500: (-80, +80),
                850: (-50, +50),
                925: (-40, +40)
            },
            'v': {
                200: (-60.0, 60.0),
                500: (-50, +50),
                850: (-40, +40),
                925: (-30, +30)
            }
        }
        self.pressure_levels = [925, 850, 500, 200]
        # --------------------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq_len = len(sample['era5_paths'])

        track = torch.tensor(sample['track_input'], dtype=torch.float32)  # (seq_len, 2)

        target = torch.tensor(
            list(zip(sample['target_lats'], sample['target_lons'])),
            dtype=torch.float32
        )

        timestamps = sample['timestamps']
        t_days = []
        t_hours = []
        for ts_str in timestamps:
            dt = datetime.datetime.strptime(ts_str, '%Y%m%d%H')
            t_days.append([dt.timetuple().tm_yday - 1])
            t_hours.append([dt.hour])

        t_day_tensor = torch.tensor(t_days, dtype=torch.float32)
        t_hour_tensor = torch.tensor(t_hours, dtype=torch.float32)

        era5_seq = []
        for path in sample['era5_paths']:
            ds = xr.open_dataset(path)
            variables = []

            for var in self.config.era5_variables_single:
                data = ds[var].values
                if var == 'sst':
                    data = np.nan_to_num(data, nan=0.0)
                v_min, v_max = self.era5_norm_range[var]
                data = (data - v_min) / (v_max - v_min)
                data = np.clip(data, 0, 1)
                data = data[np.newaxis, :, :]
                variables.append(data)

            for var in self.config.era5_variables_pres:
                var_data = ds[var].values
                normalized_layers = []
                for i, level in enumerate(self.pressure_levels):
                    layer_data = var_data[i, :, :]
                    v_min, v_max = self.era5_norm_range[var][level]
                    layer_data = (layer_data - v_min) / (v_max - v_min)
                    layer_data = np.clip(layer_data, 0, 1)
                    normalized_layers.append(layer_data[np.newaxis, :, :])
                var_normalized = np.concatenate(normalized_layers, axis=0)
                variables.append(var_normalized)

            era5_seq.append(np.concatenate(variables, axis=0))
        era5_seq = torch.tensor(np.stack(era5_seq, axis=0), dtype=torch.float32)
        # --------------------------------------------------------------------------------

        himawari_seq = []
        for path in sample['himawari_paths']:
            ds = xr.open_dataset(path)
            channels = []
            for ch in self.config.himawari_channels:
                var_data = ds[ch]
                if var_data.isnull().any().item():
                    original_lat = var_data['latitude'].to_index()
                    if not original_lat.is_monotonic_increasing:
                        if original_lat.is_monotonic_decreasing:
                            var_sorted = var_data.reindex(latitude=original_lat[::-1])
                        else:
                            var_sorted = var_data.sortby('latitude')
                    else:
                        var_sorted = var_data
                    filled_data = var_sorted.interpolate_na(dim='latitude', method='nearest')
                    filled_data = filled_data.interpolate_na(dim='longitude', method='nearest')
                    filled_data = filled_data.reindex(latitude=original_lat)
                    if filled_data.isnull().any().item():
                        filled_data = filled_data.fillna(0)
                    channels.append(filled_data.values)
                else:
                    channels.append(var_data.values)
            himawari_seq.append(np.stack(channels, axis=0))
        himawari_seq = torch.tensor(np.stack(himawari_seq, axis=0), dtype=torch.float32)

        return {
            'track': track,
            'target': target,
            'era5': era5_seq,
            'himawari': himawari_seq,
            't_day': t_day_tensor,
            't_hour': t_hour_tensor,
            'typhoon_name': sample['typhoon_name'],
            'typhoon_id': sample['typhoon_id'],
            'timestamps': sample['timestamps']
        }


def visualize_dataset_sample(sample):
    track = sample['track'].numpy()
    target = sample['target'].numpy()
    himawari = sample['himawari'].numpy()
    era5 = sample['era5'].numpy()
    seq_len = track.shape[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(track[:, 1], track[:, 0], 'o-', color='blue', label='history track')

    for idx, (lat, lon) in enumerate(target):
        label = 'forecast points' if idx == 0 else None
        ax.plot(lon, lat, 'x', color='red', label=label, markersize=8)

    for i, (lat, lon) in enumerate(track):
        ax.text(lon + 0.2, lat + 0.1, f't{i}', fontsize=8)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f"Typhoon {sample['typhoon_name']} (ID: {sample['typhoon_id']})")
    ax.legend()
    plt.grid()
    plt.show()

    for ch_idx in range(himawari.shape[1]):
        n = seq_len
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten()

        for i in range(n):
            axes[i].imshow(himawari[i, ch_idx], cmap='gray')
            axes[i].set_title(f"t{i}", fontsize=10)
            axes[i].set_axis_off()

        for i in range(n, len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle(f"Himawari Channel {ch_idx}", y=1.00)  # 总标题
        plt.tight_layout()
        plt.show()

    for ch_idx in range(era5.shape[1]):
        n = seq_len
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten()

        for i in range(n):
            im = axes[i].imshow(era5[i, ch_idx], cmap='viridis')
            axes[i].set_title(f"t{i}", fontsize=10)
            axes[i].set_axis_off()

        for i in range(n, len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle(f"ERA5 Variable {ch_idx}", y=1.00)
        plt.tight_layout()
        plt.show()


def get_typhoon_statistics(data_loader: TyphoonDataLoader):
    return data_loader.typhoon_stats


if __name__ == '__main__':
    from config import get_args
    args = get_args()
    set_seed(args.seed)
    config = TyphoonDataConfig(args)
    data_loader = TyphoonDataLoader(config, seed=args.seed)

    train_samples, val_samples, test_samples = data_loader.split_dataset(
        train_years=args.train_years,
        test_years=args.test_years
    )

    test_dataset = TyphoonDataset(test_samples, config)

    if len(test_dataset) > 0:
        sample = test_dataset[500]
        print(f"\n【测试样本详情】")
        print(f"时间信息:{sample['timestamps']}")
        # print(f"天数信息:{sample['t_day']}")
        # print(f"小时信息:{sample['t_hour']}")
        print(f"样本信息: 台风ID={sample['typhoon_id']}, 名称={sample['typhoon_name']}")
        print(f"样本数据结构: {sample.keys()}")
        print(f"轨迹数据形状: {sample['track'].shape}")
        print(f"轨迹数据：{sample['track']}")
        print(f"ERA5数据形状: {sample['era5'].shape}")
        print(f"Himawari数据形状: {sample['himawari'].shape}")
        print(f"target数据形状: {sample['target'].shape}")  # 应输出 (pred_offset, 2)

        # visualize_dataset_sample(sample)
    else:
        print("测试集为空，无法可视化")