import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEG_LEN = 1024
SNR_DEFAULT = 20
BATCH_SIZE = 32
NUM_WORKERS = 8
MAX_SEGMENTS_PER_FILE = 250
UNKNOWN_CLASS_GROUP = []

DOMAINS = {

    "source": {
        "name": "N09_M07_F10",
        "noise_enable": True,
        "snr": 0,
        "role": "labeled_source",
        "exclude_classes": [],
    },

    "test": {
        "name": "N15_M01_F10",
        "noise_enable": True,
        "snr": 0,
        "role": "target",
        "exclude_classes": [],
    },

    "source1": {
        "name": "N15_M07_F10",
        "noise_enable": True,
        "snr": 0,
        "role": "unlabeled_source",
        "exclude_classes": [],
    },

    "source2": {
        "name": "N15_M07_F04",
        "noise_enable": True,
        "snr": 0,
        "role": "unlabeled_source",
        "exclude_classes": [],
    },
}

def extract_signal(filepath):
    mat = loadmat(filepath, squeeze_me=True, struct_as_record=False)
    top_key = [k for k in mat.keys() if not k.startswith("__")][0]
    obj = mat[top_key]
    data = np.asarray(obj.Y[6].Data).squeeze().ravel().astype(np.float32)
    return data

def segment_signal(x, seg_len=1024):
    n_seg = len(x) // seg_len
    if n_seg == 0:
        return np.empty((0, seg_len), dtype=np.float32)
    x = x[: n_seg * seg_len]
    return x.reshape(n_seg, seg_len)

def add_noise_vectorized(x, snr_db):
    power = torch.mean(x ** 2, dim=-1, keepdim=True)
    noise_std = torch.sqrt(power / (10 ** (snr_db / 10)))
    noise = torch.randn_like(x) * noise_std
    return x + noise

def build_pt_from_mat(domain_name, exclude_classes=None):
    folder = os.path.join(BASE_DIR, domain_name)
    save_path = os.path.join(BASE_DIR, f"{domain_name}.pt")
    if os.path.exists(save_path):
        print(f"已存在 {save_path}，跳过生成。")
        return save_path

    exclude_set = set(exclude_classes) if exclude_classes is not None else set()

    all_segments, all_labels = [], []
    label_map = {}

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".mat"):
            continue

        label_name = os.path.splitext(fname)[0]

        if label_name in exclude_set:
            continue

        mat_path = os.path.join(folder, fname)

        data = extract_signal(mat_path)
        segs = segment_signal(data, SEG_LEN)
        if segs.shape[0] == 0:
            continue
        if segs.shape[0] > MAX_SEGMENTS_PER_FILE:
            idx = np.random.choice(segs.shape[0], MAX_SEGMENTS_PER_FILE, replace=False)
            segs = segs[idx]

        if label_name not in label_map:
            label_map[label_name] = len(label_map)
        label_id = label_map[label_name]

        all_segments.append(torch.tensor(segs, dtype=torch.float32))
        all_labels.append(torch.full((segs.shape[0],), label_id, dtype=torch.long))

    if not all_segments:
        raise RuntimeError(f"未在 {domain_name} 找到任何有效的 .mat 文件（或全部被排除）。")

    x = torch.cat(all_segments)
    y = torch.cat(all_labels)
    torch.save({"x": x, "y": y, "labels": label_map}, save_path)
    print(f" 保存 {domain_name}: {x.shape} 段, {len(label_map)} 类 -> {save_path}")
    return save_path

# ==========================================================
# 优化版 Dataset
# ==========================================================
class PUDataset(Dataset):
    def __init__(self, domain_cfg):
        self.domain_cfg = domain_cfg
        self.domain_name = domain_cfg["name"]
        self.noise_enable = domain_cfg["noise_enable"]
        self.snr = domain_cfg["snr"]
        self.data_path = os.path.join(BASE_DIR, f"{self.domain_name}.pt")

        self.exclude_classes = domain_cfg.get("exclude_classes", [])

        if not os.path.exists(self.data_path):
            build_pt_from_mat(self.domain_name, exclude_classes=self.exclude_classes)

        cache = torch.load(self.data_path)
        self.x = cache["x"]
        self.y = cache["y"]

        self.x = self.x.share_memory_()
        self.y = self.y.share_memory_()

        if self.noise_enable:
            print(f"⚙ {self.domain_name} 设置为带噪声模式 (SNR={self.snr} dB)")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        if self.noise_enable:
            x = add_noise_vectorized(x.unsqueeze(0), self.snr).squeeze(0)

        x = (x - x.mean()) / (x.std() + 1e-8)
        return x.unsqueeze(0), y  # [1,1024]


class PUDataManager:
    def __init__(self, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.domains = {}

    def build_domain(self, key):
        cfg = DOMAINS[key]
        ds = PUDataset(cfg)
        dl = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=(key == "source"),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            drop_last=True,
            prefetch_factor=4,  #
        )
        self.domains[key] = (ds, dl)
        return dl

    def get_loaders(self):
        loaders = {}
        for key in DOMAINS.keys():
            loaders[key] = self.build_domain(key)
        return loaders


if __name__ == "__main__":
    manager = PUDataManager(batch_size=BATCH_SIZE)
    loaders = manager.get_loaders()

    for key, loader in loaders.items():
        x, y = next(iter(loader))
        print(f"{key:>6}: {x.shape}, {y.shape}")