"""
Hand Temporal Model (Macro Stage)
────────────────────────────────────────────────
- Curriculum Masking: epoch 진행에 따라 연속 마스킹 길이 증가
- Mask Loss Weighting: 마스킹 구간에 높은 가중치
- SinusoidalPE: 고정 길이 제한 없음
학습: HO3D train + InterHand train
검증: InterHand val
"""

import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from smplx import MANO


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모델
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class HandTemporalModel(nn.Module):
    def __init__(self, pose_dim=45, hidden=256, n_heads=4, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(pose_dim, hidden)
        self.pos_enc    = SinusoidalPE(hidden)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads,
            dim_feedforward=512, dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden, pose_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.output_proj(x)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 데이터셋
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HandPoseDataset(Dataset):
    def __init__(self, npz_paths, seq_len=31,
                 noise_std=0.02, spike_std=0.3,
                 is_train=True, max_mask_ratio=0.5):
        """
        max_mask_ratio : 최대 마스킹 비율 (epoch 마지막에 도달)
        is_train=True  : 매 epoch 랜덤 augmentation
        is_train=False : idx 기반 고정 노이즈 (val용)
        """
        if isinstance(npz_paths, str):
            npz_paths = [npz_paths]

        self.seq_len            = seq_len
        self.noise_std          = noise_std
        self.spike_std          = spike_std
        self.is_train           = is_train
        self.max_mask_ratio     = max_mask_ratio
        self.current_mask_ratio = 0.0   # set_epoch()으로 갱신
        self.samples            = []

        for path in npz_paths:
            data        = np.load(path)
            poses       = data['hand_pose'].astype(np.float32)
            seq_lengths = data['seq_lengths']

            start = 0
            for L in seq_lengths:
                end = start + L
                for i in range(start, end - seq_len + 1):
                    self.samples.append(poses[i: i + seq_len])
                start = end

        print(f"  샘플 수: {len(self.samples):,}")

    def set_epoch(self, epoch, total_epochs):
        """Curriculum masking: epoch에 따라 마스킹 비율 선형 증가."""
        self.current_mask_ratio = ((epoch + 1) / total_epochs) * self.max_mask_ratio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gt    = self.samples[idx].copy()
        noisy = gt.copy()
        rng   = np.random.default_rng() if self.is_train \
                else np.random.default_rng(seed=idx)

        # 1. jitter
        noisy += rng.normal(0, self.noise_std, noisy.shape).astype(np.float32)

        # 2. spike
        n_spike   = rng.integers(1, 4)
        spike_idx = rng.choice(self.seq_len, size=int(n_spike), replace=False)
        for si in spike_idx:
            noisy[si] += rng.normal(0, self.spike_std, 45).astype(np.float32)

        # 3. 연속 마스킹 (curriculum)
        max_mask_len = int(self.seq_len * self.current_mask_ratio)
        mask_start, mask_len = 0, 0

        if max_mask_len >= 1:
            mask_len   = int(rng.integers(1, max_mask_len + 1))
            mask_start = int(rng.integers(0, self.seq_len - mask_len + 1))
            noisy[mask_start: mask_start + mask_len] = 0.0

        return {
            "input"     : torch.from_numpy(noisy),
            "gt"        : torch.from_numpy(gt),
            "mask_start": torch.tensor(mask_start, dtype=torch.long),
            "mask_len"  : torch.tensor(mask_len,   dtype=torch.long),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Loss
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_loss(pred, gt, mask_start, mask_len, mano_layer, device,
                 w_mask=2.0, w_accel=0.1, w_vel=0.01):
    B, T, _ = pred.shape

    weight = torch.ones(B, T, 1, device=device)
    for b in range(B):
        s = mask_start[b].item()
        l = mask_len[b].item()
        if l > 0:
            weight[b, s: s + l] = w_mask

    # 1. pose MSE
    loss_pose = (((pred - gt) ** 2) * weight).mean()

    # 2. pose velocity loss
    pred_vel = pred[:, 1:] - pred[:, :-1]
    gt_vel   = gt[:, 1:] - gt[:, :-1]
    loss_vel = torch.mean((pred_vel - gt_vel) ** 2)

    # 3. pose acceleration loss
    pred_accel = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    gt_accel   = gt[:, 2:] - 2 * gt[:, 1:-1] + gt[:, :-2]
    loss_accel = torch.mean((pred_accel - gt_accel) ** 2)

    total = loss_pose + w_accel * loss_accel + w_vel * loss_vel
    return total, loss_pose, loss_accel, loss_vel

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 학습
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train(
    train_paths,
    val_path,
    save_path       = "hand_temporal_model.pth",
    seq_len         = 31,
    epochs          = 50,
    batch_size      = 32,
    lr              = 1e-4,
    max_mask_ratio  = 0.5,
    w_mask          = 2.0,
    w_accel         = 0.1,
    w_vel           = 0.01,
    mano_path       = "mano",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    # 데이터
    print("\n[train]")
    train_dataset = HandPoseDataset(
        train_paths, seq_len=seq_len, is_train=True,
        max_mask_ratio=max_mask_ratio
    )
    print("[val]")
    val_dataset = HandPoseDataset(
        val_path, seq_len=seq_len, is_train=False,
        max_mask_ratio=max_mask_ratio
    )
    # val은 중간 난이도로 고정
    val_dataset.set_epoch(epochs // 2, epochs)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    model      = HandTemporalModel().to(device)
    mano_layer = MANO(model_path=mano_path, is_rhand=True,
                      use_pca=False).to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = torch.cuda.is_available()
    scaler  = GradScaler("cuda", enabled=use_amp)

    best_val = float('inf')
    header = (f"{'Epoch':>6}  {'Train':>10}  {'Val':>10}  "
              f"{'pose':>8}  {'accel':>8}  {'vel':>8}  {'mask_r':>7}")
    print(f"\n{'─'*len(header)}")
    print(header)
    print(f"{'─'*len(header)}")

    for epoch in range(epochs):

        # curriculum masking 갱신
        train_dataset.set_epoch(epoch, epochs)
        mask_ratio = train_dataset.current_mask_ratio

        # ── train ──────────────────────────────────────────
        model.train()
        total_train = 0.0

        for batch in train_loader:
            x          = batch["input"].to(device)
            gt         = batch["gt"].to(device)
            mask_start = batch["mask_start"].to(device)
            mask_len   = batch["mask_len"].to(device)

            optimizer.zero_grad()

            with autocast("cuda", enabled=use_amp):
                pred = model(x)
                loss, _, _, _ = compute_loss(
                    pred, gt, mask_start, mask_len,
                    mano_layer, device, w_mask, w_accel, w_vel
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_train += loss.item()

        # ── val ────────────────────────────────────────────
        model.eval()
        total_val = 0.0
        sum_lp = sum_la = sum_lv = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x          = batch["input"].to(device)
                gt         = batch["gt"].to(device)
                mask_start = batch["mask_start"].to(device)
                mask_len   = batch["mask_len"].to(device)

                with autocast("cuda", enabled=use_amp):
                    pred = model(x)
                    val_loss, lp, la, lv = compute_loss(
                        pred, gt, mask_start, mask_len,
                        mano_layer, device, w_mask, w_accel, w_vel
                    )

                total_val += val_loss.item()
                sum_lp    += lp.item()
                sum_la    += la.item()
                sum_lv    += lv.item()

        scheduler.step()

        n         = len(val_loader)
        avg_train = total_train / len(train_loader)
        avg_val   = total_val   / n

        saved = ""
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), save_path)
            saved = "  ← 저장"

        print(f"{epoch+1:>6}  {avg_train:>10.5f}  {avg_val:>10.5f}  "
              f"{sum_lp/n:>8.5f}  {sum_la/n:>8.5f}  {sum_lv/n:>8.5f}  "
              f"{mask_ratio:>7.3f}{saved}")

    print(f"\n학습 완료 | best val: {best_val:.6f} | 저장: {save_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 추론
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TemporalSmoother:
    def __init__(self, model_path, seq_len=31, device=None):
        self.device  = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.seq_len = seq_len
        self.buffer  = []

        self.model = HandTemporalModel().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()

    def update(self, pose_tensor):
        pose_np = (pose_tensor.cpu().numpy().flatten()
                   if pose_tensor is not None
                   else np.zeros(45, dtype=np.float32))

        self.buffer.append(pose_np)
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

        if len(self.buffer) < self.seq_len:
            return pose_tensor

        seq = np.stack(self.buffer, axis=0)
        x   = torch.tensor(seq).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(x)

        return pred[0, self.seq_len // 2].unsqueeze(0)

    def process_video(self, pose_list):
        """전체 영상을 한 번에 처리 (오프라인).
        pad_start / pad_end 방식으로 항상 중앙 프레임 = i 프레임 보장.
        """
        half = self.seq_len // 2
        N    = len(pose_list)

        # None 처리 후 numpy 변환
        poses = []
        for p in pose_list:
            if p is None:
                poses.append(np.zeros(45, dtype=np.float32))
            else:
                poses.append(p.cpu().numpy().flatten().astype(np.float32))
        poses = np.stack(poses, axis=0)   # (N, 45)

        # 앞뒤 패딩: 첫/마지막 프레임 반복
        pad_start = np.repeat(poses[:1],  half, axis=0)   # (half, 45)
        pad_end   = np.repeat(poses[-1:], half, axis=0)   # (half, 45)
        padded    = np.concatenate([pad_start, poses, pad_end], axis=0)  # (N+seq_len-1, 45)

        results = []
        for i in range(N):
            window = padded[i: i + self.seq_len]   # (seq_len, 45) 항상 중앙=i
            x      = torch.tensor(window).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred = self.model(x)

            results.append(pred[0, half].unsqueeze(0))

        return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 진입점
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU :", torch.cuda.get_device_name(0))

    train(
        train_paths    = [
            "ho3d_train.npz",
            "interhand_train.npz",
        ],
        val_path       = "interhand_val.npz",
        save_path      = "hand_temporal_model_macro.pth",
        seq_len        = 31,
        epochs         = 50,
        batch_size     = 32,
        lr             = 1e-4,
        max_mask_ratio = 0.5,   # 마지막 epoch에 seq_len의 50% 마스킹
        w_mask         = 2.0,   # 마스킹 구간 loss 2배
        w_accel        = 0.5,
        w_vel          = 0.1,
        mano_path      = "mano",
    )