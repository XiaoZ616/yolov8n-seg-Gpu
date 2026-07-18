## YOLOv8n-seg 的网络结构主要由三部分组成
    Backbone (骨干网络)：负责从输入图片中提取基础特征。

    Neck (颈部网络)：负责融合不同尺度的特征，让模型能“看”清大小不同的目标。

    Head (头部网络)：负责最终的任务预测，对于分割任务来说，就是生成目标框和分割掩码。

## Backbone
    YOLOv8n的Backbone由以下部分顺序组成：
        一个Stem卷积（3x3, stride=2）—— 初步下采样
        
        4个Stage，每个Stage包含：
            一个下采样卷积（3x3, stride=2）
            一个C2f模块（C2f内部包含若干个Bottleneck）
            （最后一个Stage还额外跟了一个SPPF模块）

        具体参数（nano版本）：
            Stage1: C2f的 n=3
            Stage2: C2f的 n=6
            Stage3: C2f的 n=6
            Stage4: C2f的 n=3 + SPPF

## 损失函数 loss.py
YOLOv8 分割损失由三部分组成：

边界框损失（Box Loss）：使用 CIoU（Complete IoU），衡量预测框和真实框的重合度、中心点距离和长宽比。

分类损失（Cls Loss）：使用 BCEWithLogitsLoss（二分类交叉熵），判断每个锚点属于哪个类别。

分割损失（Seg Loss）：使用 Dice Loss + BCE Loss 的组合，衡量预测掩码和真实掩码的像素级重合度。

