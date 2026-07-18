"""
训练脚本：极速版，暂时关闭分割损失，优先训练检测分支
每轮预计 1~2 分钟（RTX 4060）
"""

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True  # 自动优化卷积

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import cv2
import numpy as np
import yaml
from tqdm import tqdm

from model import YOLOv8Seg
from Loss import YOLOv8SegLoss

# ======================== 配置区域 ========================
DATASET_YAML = "dataset/dataset.yaml"
EPOCHS = 50
BATCH_SIZE = 8           # 降低 batch size 加速单次迭代
IMGSZ = 640
LR = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path("runs/my_train")
NUM_CLASSES = 4
NUM_MASKS = 32
TOP_K = 5                # 极速：只保留 5 个候选框
SEG_GAIN = 0.0           # 【改动1】暂时关闭分割损失，先让检测分支学习
# ==========================================================

# ==================== 数据集（预加载） ====================
class YOLOSegDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640):
        self.img_size = img_size
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        all_files = sorted(list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png")))
        self.img_files = [f for f in all_files if (self.label_dir / f"{f.stem}.txt").exists()]
        
        self.imgs = []
        self.boxes = []
        self.classes = []
        self.masks = []
        print(f"⏳ 预加载 {len(self.img_files)} 张图片...")
        for f in tqdm(self.img_files, desc="预加载"):
            img, boxes, cls, mask = self._load_one(f)
            self.imgs.append(img)
            self.boxes.append(boxes)
            self.classes.append(cls)
            self.masks.append(mask)
        print("✅ 预加载完成！")
        
    def _load_one(self, img_path):
        label_path = self.label_dir / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        boxes, classes, all_points = [], [], []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = list(map(float, line.strip().split()))
                    if len(parts) < 7: continue
                    cls_id = int(parts[0])
                    points = np.array(parts[1:]).reshape(-1, 2)
                    points[:, 0] = points[:, 0] * w_orig
                    points[:, 1] = points[:, 1] * h_orig
                    x1 = points[:, 0].min(); y1 = points[:, 1].min()
                    x2 = points[:, 0].max(); y2 = points[:, 1].max()
                    x1 = x1 / w_orig * self.img_size
                    y1 = y1 / h_orig * self.img_size
                    x2 = x2 / w_orig * self.img_size
                    y2 = y2 / h_orig * self.img_size
                    boxes.append([x1, y1, x2, y2])
                    classes.append(cls_id)
                    all_points.append(points)
        
        mask_h, mask_w = self.img_size // 8, self.img_size // 8
        if len(boxes) == 0:
            return img_tensor, torch.zeros((0, 4)), torch.zeros((0, 1)), torch.zeros((1, mask_h, mask_w))
        
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        classes_t = torch.tensor(classes, dtype=torch.long)
        mask_tensor = np.zeros((len(boxes), mask_h, mask_w), dtype=np.float32)
        scale_x, scale_y = mask_w / self.img_size, mask_h / self.img_size
        for i, pts in enumerate(all_points):
            pts_scaled = pts * np.array([scale_x, scale_y])
            mask_single = np.zeros((mask_h, mask_w), dtype=np.uint8)
            pts_int = pts_scaled.reshape(-1, 1, 2).astype(np.int32)
            cv2.fillPoly(mask_single, [pts_int], 1)
            mask_tensor[i] = mask_single
        mask_t = torch.from_numpy(mask_tensor).float()
        return img_tensor, boxes_t, classes_t, mask_t

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.boxes[idx], self.classes[idx], self.masks[idx]

# ==================== 解码（极简版，只取 top-k 候选） ====================
def decode_outputs(detections, proto, img_size=640, num_classes=4, top_k=5):
    """
    极简解码：只保留 top_k 个最高置信度的候选框及其掩码系数
    proto 已被下采样到 80x80（在外部完成）
    """
    out = detections[0]  # [B, C, 80, 80]
    B, C, H, W = out.shape
    nm = proto.shape[1]          # 32
    npr = 32
    box_out = out[:, :64, :, :]
    cls_out = out[:, 64:64+num_classes, :, :]
    mask_out = out[:, 64+num_classes:, :, :]  # [B, nm*npr, H, W]

    batch_boxes, batch_cls, batch_masks = [], [], []
    for b in range(B):
        cls_scores = cls_out[b].sigmoid()          # [num_classes, H, W]
        max_scores, max_idxs = cls_scores.max(dim=0)  # [H, W]
        flat_scores = max_scores.flatten()
        flat_idxs = max_idxs.flatten()
        
        k = min(top_k, flat_scores.numel())
        top_vals, top_indices = torch.topk(flat_scores, k)
        valid = top_vals > 0.1
        top_indices = top_indices[valid]
        if len(top_indices) == 0:
            batch_boxes.append(torch.zeros((0, 4), device=out.device))
            batch_cls.append(torch.zeros((0, num_classes), device=out.device))
            batch_masks.append(torch.zeros((0, H, W), device=out.device))
            continue
        
        ys = top_indices // W
        xs = top_indices % W
        
        cls_logits = cls_out[b, :, ys, xs].permute(1, 0)      # [N, num_classes]
        box_preds = box_out[b, 0:4, ys, xs].permute(1, 0).sigmoid()  # [N, 4]
        
        # 掩码系数：每个位置是 [nm*npr]，reshape 为 [nm, npr]，再对 npr 取平均
        coeffs = mask_out[b, :, ys, xs].permute(1, 0)          # [N, nm*npr]
        coeffs = coeffs.view(-1, nm, npr)                      # [N, nm, npr]
        coeffs_mean = coeffs.mean(dim=2)                       # [N, nm]
        
        # 与 proto 相乘得到掩码（proto 已在外部下采样到 HxW）
        proto_b = proto[b]                                     # [nm, H, W]
        proto_flat = proto_b.view(nm, -1)                      # [nm, H*W]
        masks_flat = torch.matmul(coeffs_mean, proto_flat)     # [N, H*W]
        pred_masks_logits = masks_flat.view(-1, H, W)          # [N, H, W]
        
        batch_boxes.append(box_preds)
        batch_cls.append(cls_logits)
        batch_masks.append(pred_masks_logits)
    
    return batch_boxes, batch_cls, batch_masks

# ==================== 训练主函数 ====================
def main():
    print("=" * 60)
    print("🐟 训练（极速版，检测优先）")
    print(f"💻 设备: {DEVICE}")
    print(f"📦 Batch Size: {BATCH_SIZE}")
    print(f"🔢 Top-K: {TOP_K}")
    print(f"🔧 分割损失权重: {SEG_GAIN} (暂时关闭)")
    print("=" * 60)
    
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATASET_YAML, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    
    train_img_dir = Path(data_cfg['path']) / data_cfg['train']
    train_label_dir = Path(data_cfg['path']) / "labels" / "train"
    val_img_dir = Path(data_cfg['path']) / data_cfg['val']
    val_label_dir = Path(data_cfg['path']) / "labels" / "val"
    
    train_dataset = YOLOSegDataset(train_img_dir, train_label_dir, IMGSZ)
    val_dataset = YOLOSegDataset(val_img_dir, val_label_dir, IMGSZ)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False   # 关闭 pin_memory 减少开销
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"📊 训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张")
    
    model = YOLOv8Seg(num_classes=NUM_CLASSES, num_masks=NUM_MASKS).to(DEVICE)
    print(f"🧠 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    # 【改动2】使用配置的分割损失权重
    criterion = YOLOv8SegLoss(box_gain=7.5, cls_gain=0.5, seg_gain=SEG_GAIN)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, target_boxes, target_cls, target_masks in progress:
            imgs = imgs.to(DEVICE)
            target_boxes = [b.to(DEVICE) for b in target_boxes]
            target_cls = [c.to(DEVICE) for c in target_cls]
            target_masks = [m.to(DEVICE) for m in target_masks]
            
            # 前向
            detections, proto = model(imgs)
            
            # 将 proto 下采样到 80x80（与 detections[0] 尺寸一致）—— 每个 batch 只做一次
            H, W = detections[0].shape[2], detections[0].shape[3]
            proto_down = torch.nn.functional.interpolate(proto, size=(H, W), mode='bilinear', align_corners=False)
            
            # 解码
            pred_boxes, pred_cls, pred_masks = decode_outputs(
                detections, proto_down, IMGSZ, NUM_CLASSES, top_k=TOP_K
            )
            
            valid_boxes = [b for b in pred_boxes if b.numel() > 0]
            if not valid_boxes:
                continue
            all_pred_boxes = torch.cat(valid_boxes, dim=0)
            all_pred_cls = torch.cat([c for c in pred_cls if c.numel() > 0], dim=0)
            all_pred_masks = torch.cat([m for m in pred_masks if m.numel() > 0], dim=0)
            
            all_target_boxes = torch.cat([b for b in target_boxes if b.numel() > 0], dim=0)
            all_target_cls = torch.cat([c for c in target_cls if c.numel() > 0], dim=0)
            all_target_masks = torch.cat([m for m in target_masks if m.numel() > 0], dim=0)
            
            if all_target_boxes.numel() == 0:
                continue
            
            # 【改动3】关键：只有预测数量和真实数量相等时才训练，避免错误匹配
            if all_pred_boxes.size(0) != all_target_boxes.size(0):
                continue  # 跳过此 batch
            
            n_match = all_pred_boxes.size(0)  # 此时两者相等
            
            loss, box_loss, cls_loss, seg_loss = criterion(
                all_pred_boxes[:n_match],
                all_pred_cls[:n_match],
                all_pred_masks[:n_match],
                all_target_boxes[:n_match],
                torch.nn.functional.one_hot(all_target_cls[:n_match], num_classes=NUM_CLASSES).float(),
                all_target_masks[:n_match]
            )
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress.set_postfix({
                'loss': loss.item(),
                'box': box_loss.item(),
                'cls': cls_loss.item(),
                'seg': seg_loss.item() if seg_loss is not None else 0.0
            })
        
        scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"📉 平均训练损失: {avg_loss:.4f}")
        
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0
            val_count = 0
            with torch.no_grad():
                for imgs, target_boxes, target_cls, target_masks in val_loader:
                    imgs = imgs.to(DEVICE)
                    target_boxes = [b.to(DEVICE) for b in target_boxes]
                    target_cls = [c.to(DEVICE) for c in target_cls]
                    target_masks = [m.to(DEVICE) for m in target_masks]
                    
                    detections, proto = model(imgs)
                    H, W = detections[0].shape[2], detections[0].shape[3]
                    proto_down = torch.nn.functional.interpolate(proto, size=(H, W), mode='bilinear', align_corners=False)
                    pred_boxes, pred_cls, pred_masks = decode_outputs(
                        detections, proto_down, IMGSZ, NUM_CLASSES, top_k=TOP_K
                    )
                    
                    valid_boxes = [b for b in pred_boxes if b.numel() > 0]
                    if not valid_boxes:
                        continue
                    all_pred_boxes = torch.cat(valid_boxes, dim=0)
                    all_pred_cls = torch.cat([c for c in pred_cls if c.numel() > 0], dim=0)
                    all_pred_masks = torch.cat([m for m in pred_masks if m.numel() > 0], dim=0)
                    
                    all_target_boxes = torch.cat([b for b in target_boxes if b.numel() > 0], dim=0)
                    all_target_cls = torch.cat([c for c in target_cls if c.numel() > 0], dim=0)
                    all_target_masks = torch.cat([m for m in target_masks if m.numel() > 0], dim=0)
                    
                    if all_target_boxes.numel() == 0:
                        continue
                    
                    # 验证中也使用相同匹配规则
                    if all_pred_boxes.size(0) != all_target_boxes.size(0):
                        continue
                    n_match = all_pred_boxes.size(0)
                    
                    loss_val, _, _, _ = criterion(
                        all_pred_boxes[:n_match],
                        all_pred_cls[:n_match],
                        all_pred_masks[:n_match],
                        all_target_boxes[:n_match],
                        torch.nn.functional.one_hot(all_target_cls[:n_match], num_classes=NUM_CLASSES).float(),
                        all_target_masks[:n_match]
                    )
                    val_loss += loss_val.item()
                    val_count += 1
            
            avg_val_loss = val_loss / max(val_count, 1)
            print(f"📊 验证损失: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, SAVE_DIR / 'best.pt')
                print(f"💾 保存最佳模型 (损失: {best_val_loss:.4f})")
    
    print("\n" + "=" * 60)
    print(f"🎉 训练完成！模型保存至: {SAVE_DIR / 'best.pt'}")
    print("=" * 60)

if __name__ == "__main__":
    main()