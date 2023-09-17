import torch.nn as nn
import torchvision.models as models

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
        self.efficientnet_images = nn.Sequential(*list(efficientnet.children())[:-3])
        
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Define additional layers for upsampling to reach 50x50 resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
            )
        
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
            

    def forward(self, x):

        # Extract features from the pre-trained ResNet-50
        efficientnet_images = self.efficientnet_images(x)
        x = x + efficientnet_images
        x = self.first_conv(x)
        x = self.encoder(x)
        x = self.decoder(x)
        output = self.final_conv(x)

        return output