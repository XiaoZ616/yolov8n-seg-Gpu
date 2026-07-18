"""
损失函数：YOLOv8 实例分割损失
包含：Box Loss (CIoU), Cls Loss (BCE), Seg Loss (Dice + BCE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==================== 1. CIoU 损失 (边界框) ====================
def ciou_loss(pred_boxes, target_boxes):
    """
    计算 CIoU 损失
    pred_boxes, target_boxes: [N, 4] (x1, y1, x2, y2) 格式
    """
    # 转换为 (x, y, w, h) 方便计算
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    target_x1, target_y1, target_x2, target_y2 = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]

    pred_w = pred_x2 - pred_x1
    pred_h = pred_y2 - pred_y1
    target_w = target_x2 - target_x1
    target_h = target_y2 - target_y1

    # IoU 计算
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    pred_area = pred_w * pred_h
    target_area = target_w * target_h
    union_area = pred_area + target_area - inter_area
    iou = inter_area / (union_area + 1e-7)

    # 中心点距离
    pred_cx = (pred_x1 + pred_x2) / 2
    pred_cy = (pred_y1 + pred_y2) / 2
    target_cx = (target_x1 + target_x2) / 2
    target_cy = (target_y1 + target_y2) / 2
    center_dist_sq = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
    
    # 对角线距离 (包围框)
    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)
    enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-7

    # v 值 (长宽比一致性)
    v = (4 / (math.pi ** 2)) * (torch.atan(pred_w / (pred_h + 1e-7)) - torch.atan(target_w / (target_h + 1e-7))) ** 2
    alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - (center_dist_sq / enclose_diag_sq) - alpha * v
    return 1 - ciou

# ==================== 2. 分割损失 (Dice + BCE) ====================
class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_masks, target_masks):
        """
        pred_masks: [N, H, W]  (预测的掩码 logits)
        target_masks: [N, H, W] (真实的掩码，0/1)
        """
        # 展平
        pred_flat = pred_masks.view(pred_masks.size(0), -1)
        target_flat = target_masks.view(target_masks.size(0), -1)
        
        # Dice Loss
        intersection = (pred_flat.sigmoid() * target_flat).sum(dim=1)
        union = pred_flat.sigmoid().sum(dim=1) + target_flat.sum(dim=1)
        dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
        dice_loss = 1 - dice.mean()
        
        # BCE Loss
        bce_loss = self.bce(pred_flat, target_flat)
        
        return dice_loss + bce_loss

# ==================== 3. 组合损失 ====================
class YOLOv8SegLoss(nn.Module):
    def __init__(self, box_gain=7.5, cls_gain=0.5, seg_gain=0.5):
        super().__init__()
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.seg_gain = seg_gain
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.seg_loss = SegLoss()

    def forward(self, pred_boxes, pred_cls, pred_masks, target_boxes, target_cls, target_masks):
        """
        参数:
            pred_boxes: [N, 4] 预测框 (x1, y1, x2, y2)
            pred_cls: [N, num_classes] 预测类别 logits
            pred_masks: [N, H, W] 预测掩码 logits
            target_boxes: [N, 4] 真实框
            target_cls: [N, num_classes] 真实类别 (one-hot)
            target_masks: [N, H, W] 真实掩码 (0/1)
        """
        # 1. 框损失
        box_loss = ciou_loss(pred_boxes, target_boxes).mean()
        
        # 2. 分类损失
        cls_loss = self.cls_loss(pred_cls, target_cls)
        
        # 3. 分割损失
        seg_loss = self.seg_loss(pred_masks, target_masks)
        
        # 加权求和
        total_loss = self.box_gain * box_loss + self.cls_gain * cls_loss + self.seg_gain * seg_loss
        return total_loss, box_loss, cls_loss, seg_loss