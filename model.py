"""
完整模型文件：YOLOv8n-seg 实例分割网络
将 Backbone、Neck、Head 组装成端到端的模型

用法:
    from model import YOLOv8Seg
    model = YOLOv8Seg(num_classes=4, num_masks=32)
    output, proto = model(images)
"""

import torch
import torch.nn as nn

# ======================== 以下为 Backbone 部分 =========================

def autopad(k, p=None, d=1):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

class CSPDarknet(nn.Module):
    """YOLOv8 骨干网络 (nano 配置)"""
    def __init__(self):
        super().__init__()
        self.stem = Conv(3, 16, k=3, s=2)
        self.stage1 = nn.Sequential(
            Conv(16, 32, k=3, s=2),
            C2f(32, 32, n=1, shortcut=True)
        )
        self.stage2 = nn.Sequential(
            Conv(32, 64, k=3, s=2),
            C2f(64, 64, n=2, shortcut=True)
        )
        self.stage3 = nn.Sequential(
            Conv(64, 128, k=3, s=2),
            C2f(128, 128, n=2, shortcut=True)
        )
        self.stage4 = nn.Sequential(
            Conv(128, 256, k=3, s=2),
            C2f(256, 256, n=1, shortcut=True),
            SPPF(256, 256, k=5)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        p3 = x   # 1/8, 64ch
        x = self.stage3(x)
        p4 = x   # 1/16, 128ch
        x = self.stage4(x)
        p5 = x   # 1/32, 256ch
        return p3, p4, p5

# ======================== 以下为 Neck 部分 =========================

class PANFPN(nn.Module):
    """特征融合网络"""
    def __init__(self):
        super().__init__()
        # Top-down
        self.p5_reduce = Conv(256, 128, k=1, s=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f_p4 = C2f(128 + 128, 128, n=1, shortcut=False)

        self.p4_reduce = Conv(128, 64, k=1, s=1)
        self.c2f_p3 = C2f(64 + 64, 64, n=1, shortcut=False)

        # Bottom-up
        self.p3_down = Conv(64, 128, k=3, s=2)
        self.c2f_p4_up = C2f(128 + 128, 128, n=1, shortcut=False)

        self.p4_down = Conv(128, 256, k=3, s=2)
        self.c2f_p5_up = C2f(256 + 256, 256, n=1, shortcut=False)

    def forward(self, p3, p4, p5):
        # Top-down
        p5_ = self.p5_reduce(p5)
        p5_up = self.upsample(p5_)
        p4_cat = torch.cat([p4, p5_up], dim=1)
        p4_out = self.c2f_p4(p4_cat)

        p4_ = self.p4_reduce(p4_out)
        p4_up = self.upsample(p4_)
        p3_cat = torch.cat([p3, p4_up], dim=1)
        p3_out = self.c2f_p3(p3_cat)

        # Bottom-up
        p3_down = self.p3_down(p3_out)
        p4_cat_up = torch.cat([p3_down, p4_out], dim=1)
        p4_out = self.c2f_p4_up(p4_cat_up)

        p4_down = self.p4_down(p4_out)
        p5_cat_up = torch.cat([p4_down, p5], dim=1)
        p5_out = self.c2f_p5_up(p5_cat_up)

        return p3_out, p4_out, p5_out

# ======================== 以下为 Head 部分 =========================

class Proto(nn.Module):
    """原型掩码生成器"""
    def __init__(self, c1, nm=32):
        super().__init__()
        self.nm = nm
        self.cv1 = Conv(c1, c1 // 2, k=3)
        self.cv2 = Conv(c1 // 2, c1 // 2, k=3)
        self.cv3 = Conv(c1 // 2, nm, k=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.cv1(x)
        x = self.cv2(x)
        x = self.cv3(x)
        return self.upsample(x)

class DetectHead(nn.Module):
    """单个尺度的检测头"""
    def __init__(self, nc=4, nm=32, npr=32, ch=128):
        super().__init__()
        self.nc = nc
        self.nm = nm
        self.npr = npr
        c2 = max(16, ch // 4, self.nm * npr)
        c3 = max(ch, min(nc, 100))

        self.cv2 = nn.Sequential(
            Conv(ch, c2, 3),
            Conv(c2, c2, 3),
            nn.Conv2d(c2, 4 * 16, 1)
        )
        self.cv3 = nn.Sequential(
            Conv(ch, c3, 3),
            Conv(c3, c3, 3),
            nn.Conv2d(c3, self.nc, 1)
        )
        self.cv4 = nn.Sequential(
            Conv(ch, c2, 3),
            Conv(c2, c2, 3),
            nn.Conv2d(c2, self.nm * npr, 1)
        )

    def forward(self, x):
        box = self.cv2(x)
        cls = self.cv3(x)
        mask = self.cv4(x)
        return torch.cat([box, cls, mask], dim=1)

class SegmentHead(nn.Module):
    """组合三个尺度的检测头和原型生成器"""
    def __init__(self, nc=4, nm=32, npr=32):
        super().__init__()
        self.detect_p3 = DetectHead(nc, nm, npr, ch=64)
        self.detect_p4 = DetectHead(nc, nm, npr, ch=128)
        self.detect_p5 = DetectHead(nc, nm, npr, ch=256)
        self.proto = Proto(c1=64, nm=nm)

    def forward(self, p3, p4, p5):
        out_p3 = self.detect_p3(p3)
        out_p4 = self.detect_p4(p4)
        out_p5 = self.detect_p5(p5)
        proto = self.proto(p3)
        return [out_p3, out_p4, out_p5], proto

# ======================== 完整模型组装 =========================

class YOLOv8Seg(nn.Module):
    """
    YOLOv8 实例分割网络 (nano 版本)
    参数:
        num_classes (int): 类别数 (默认 4)
        num_masks (int): 原型掩码数量 (默认 32)
        num_proto_coeffs (int): 掩码系数的数量 (默认 32)
        device (str or torch.device): 运行设备 (默认自动检测 GPU)
    """
    def __init__(self, num_classes=4, num_masks=32, num_proto_coeffs=32, device=None):
        super().__init__()
        self.device = device if device is not None else \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.backbone = CSPDarknet()
        self.neck = PANFPN()
        self.head = SegmentHead(
            nc=num_classes,
            nm=num_masks,
            npr=num_proto_coeffs
        )

        self.to(self.device)

    def forward(self, x):
        """
        输入:
            x: [B, 3, H, W]  (建议 H, W 为 32 的倍数，如 640x640)
        返回:
            detections: list 长度为 3，每个元素 [B, C, H_i, W_i]  (检测头输出)
            proto: [B, num_masks, H/4, W/4]  (原型掩码)
        """
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)
        detections, proto = self.head(p3, p4, p5)
        return detections, proto


# ======================== 快速测试 =========================
if __name__ == "__main__":
    print("=" * 60)
    print(" 测试完整 YOLOv8Seg 模型")
    print("=" * 60)

    # 配置
    batch_size = 4
    num_classes = 4
    num_masks = 32
    input_size = 640

    # 自动检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" 使用设备: {device}")

    # 创建模型 (自动移动到 GPU)
    model = YOLOv8Seg(num_classes=num_classes, num_masks=num_masks, device=device)
    print(f" 模型创建成功，当前设备: {next(model.parameters()).device}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f" 模型总参数量: {total_params:,}")

    # 模拟输入 (放到与模型相同的设备上)
    dummy = torch.randn(batch_size, 3, input_size, input_size, device=device)
    print(f" 输入形状: {dummy.shape}, 设备: {dummy.device}")

    # 前向传播
    with torch.no_grad():
        detections, proto = model(dummy)

    print("\n 输出形状:")
    for i, out in enumerate(detections):
        print(f"  detections[{i}]: {out.shape}, 设备: {out.device}")
    print(f"  proto: {proto.shape}, 设备: {proto.device}")

    # 输出通道解析
    C = detections[0].shape[1]
    expected = 64 + num_classes + num_masks * 32
    print(f"\n 检测头输出通道数: {C} (预期 {expected})")
    assert C == expected, "通道数不匹配！"
    print(" 所有测试通过！")