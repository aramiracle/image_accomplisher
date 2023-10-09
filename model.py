import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.utils import save_image

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Define convolutional and deconvolutional blocks
        self.conv1 = self.conv_block(3, 8)
        self.conv2 = self.conv_block(8, 16)
        self.conv3 = self.conv_block(16, 32)
        self.conv4 = self.conv_block(32, 64)

        self.deconv4 = self.deconv_block(64, 32)
        self.deconv3 = self.deconv_block(64, 16)
        self.deconv2 = self.deconv_block(32, 8)
        self.deconv1 = self.deconv_block(16, 3)

        self.final_conv = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1)

    def conv_block(self, in_channels, out_channels):
        # Define a convolutional block with LeakyReLU and BatchNormalization
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_channels),
            ResidualBlock(out_channels, out_channels)
        )

    def deconv_block(self, in_channels, out_channels):
        # Define a deconvolutional block with LeakyReLU and BatchNormalization
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_channels),
            ResidualBlock(out_channels, out_channels)
        )
    
    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        # Decoder
        z4 = self.deconv4(x4)
        z3 = self.deconv3(torch.cat((z4, x3), dim=1))
        z2 = self.deconv2(torch.cat((z3, x2), dim=1))
        z1 = self.deconv1(torch.cat((z2, x1), dim=1))

        output = self.final_conv(torch.cat((z1, x), dim=1))
        
        return output
    
class EfficientNetGenerator(nn.Module):
    def __init__(self):
        super(EfficientNetGenerator, self).__init__()

        # Load the pre-trained EfficientNet model
        efficientnet = models.efficientnet_v2_s(weights='EfficientNet_V2_S_Weights.DEFAULT').eval()
        
        # Remove the fully connected layers at the end
        self.efficientnet_features = nn.Sequential(*list(efficientnet.children())[:-1])
        
        self.fc = nn.Linear(1280, 64)

        self.conv1 = self.conv_block(3, 16)
        self.conv2 = self.conv_block(16, 32)
        self.conv3 = self.conv_block(32, 64)

        self.deconv2 = self.deconv_block(64, 32)
        self.deconv1 = self.deconv_block(32, 16)

        self.up1 = self.deconv_block(3 * 16, 3)
        self.up2 = self.deconv_block(3 * 32, 16)
        self.up3 = self.deconv_block(2 * 64, 32)

        self.final_conv = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1)

    def conv_block(self, in_channels, out_channels):
        # Define a convolutional block with LeakyReLU and BatchNormalization
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=4, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_channels),
            ResidualBlock(out_channels, out_channels)
        )
    
    def deconv_block(self, in_channels, out_channels):
        # Define a deconvolutional block with LeakyReLU and BatchNormalization
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=8, stride=4, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_channels),
            ResidualBlock(out_channels, out_channels)
        )
    
    def forward(self, x):
        # Extract features from the pre-trained EfficientNet
        z = self.efficientnet_features(x)  # 1280x1x1
        z3 = self.fc(z.view(z.size(0), -1)).unsqueeze(2).unsqueeze(3)  # 64x1x1

        # UpSample
        z2 = self.deconv2(z3)  # 32x4x4
        z1 = self.deconv1(z2)  # 16x16x16
        
        # Encoder
        x1 = self.conv1(x)  # 16x16x16
        x2 = self.conv2(x1)  # 32x4x4
        x3 = self.conv3(x2)  # 64x1x1

        # Decoder
        up3 = self.up3(torch.cat((x3, z3), dim=1))  # 32x4x4
        up2 = self.up2(torch.cat((up3, x2, z2), dim=1))  # 16x32x32
        up1 = self.up1(torch.cat((up2, x1, z1), dim=1))  # 3x64x64

        output = self.final_conv(torch.cat((up1, x), dim=1))

        return output
    
# Define a Residual Block for the generator model.
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        # Define convolution layers and batch normalization.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Add the residual connection
        return out
