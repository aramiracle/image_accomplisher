import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.utils import save_image

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = self.conv_block()
        self.conv2 = self.conv_block()
        self.conv3 = self.conv_block()
        self.conv4 = self.conv_block()

    def conv_block(self):
        return nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 3, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.conv1(x) + x
        x2 = self.conv2(x1) + x
        x3 = self.conv3(x2)

        return x3
        


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(64 * 64 * 64, 1),  # Fully connected layer to output a single scalar
            nn.Sigmoid()  # Sigmoid activation to squash the output to [0, 1]
        )

    def forward(self, x):
        return self.model(x)
    
class EfficientNetGenerator(nn.Module):
    def __init__(self):
        super(EfficientNetGenerator, self).__init__()

        # Load the pre-trained ResNet-50 model
        efficientnet = models.efficientnet_b3(weights='EfficientNet_B3_Weights.DEFAULT')
        
        # Remove the fully connected layers at the end
        self.efficientnet_features = nn.Sequential(*list(efficientnet.children())[:-2])
        

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Define additional layers for upsampling to reach 50x50 resolution
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(1536, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
            
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        # Extract features from the pre-trained ResNet-50
        z = self.efficientnet_features(x) #1536x2x2


        z4 = self.deconv4(z) #256x4x4
        z3 = self.deconv3(z4) #128x8x8
        z2 = self.deconv2(z3) #64x16x16
        z1 = self.deconv1(z2) #3x32x32

        #Encoder
        x1 = self.conv1(x) #32x32x32
        x2 = self.conv2(x1) #64x16x16
        x3 = self.conv3(x2) #128x8x8
        x4 = self.conv4(x3) #256x4x4
        
        #Decoder
        up4 = self.up4(x4 + z4) #128x8x8
        up3 = self.up3(up4 + x3 + z3) #64x16x16
        up2 = self.up2(up3 + x2 + z2) #32x32x32
        up1 = self.up1(up2 + x1 + z1) #3x64x64

        return up1