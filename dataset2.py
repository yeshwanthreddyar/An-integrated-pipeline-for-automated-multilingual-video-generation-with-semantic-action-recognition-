# dataset2.py
import torch
import torch.nn as nn
import torchvision.models.video as models


class VideoClassifier(nn.Module):
    """
    Fine-tuned 3D ResNet-18 (R3D_18) model for video classification.
    Takes video input of shape [B, T, C, H, W] where:
        B = batch size
        T = temporal frames
        C = channels (3)
        H, W = height, width
    """

    def __init__(self, num_classes=101):
        super(VideoClassifier, self).__init__()

        # Load pretrained 3D ResNet (trained on Kinetics-400)
        self.backbone = models.r3d_18(weights="KINETICS400_V1")

        # Remove final classification layer
        self.backbone.fc = nn.Identity()

        # Custom pooling + fully connected layer
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))  # Pool across time, height, width
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward pass
        Input: [B, T, C, H, W]
        Output: [B, num_classes]
        """
        if x.ndim != 5:
            raise ValueError(f"Expected [B, T, C, H, W], got {x.shape}")

        # Permute to [B, C, T, H, W] as expected by Conv3D
        x = x.permute(0, 2, 1, 3, 4)

        # Backbone forward
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Global pooling + classification
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


# ✅ Quick test
if __name__ == "__main__":
    model = VideoClassifier(num_classes=101)
    dummy = torch.randn(2, 16, 3, 112, 112)  # [batch, frames, channels, height, width]
    out = model(dummy)
    print("Output shape:", out.shape)  # torch.Size([2, 101])
