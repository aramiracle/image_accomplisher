import os
import re
import shutil
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure, VisualInformationFidelity, PeakSignalNoiseRatio
from model import *
from data_loader import ImageEnhancementDataset


def lnl1_metric(prediction, target):
    max = torch.maximum(prediction, target) + 1e-3 * torch.ones_like(prediction)
    norm1 = torch.absolute(prediction - target)
    normalize_norm1 = torch.mean(torch.mul(norm1, max.pow(-1)))
    lnl1_value = -20*torch.log10(normalize_norm1)

    return lnl1_value

def PSNR_SSIM_LNL1_loss(prediction, target):
    psnr = PeakSignalNoiseRatio()
    psnr_value = psnr(prediction, target)
    psnr_loss = -psnr_value

    # Calculate Structural Similarity Index (SSIM)
    ssim = StructuralSimilarityIndexMeasure(data_range=1)
    ssim_value = ssim(prediction, target)

    # Calculate a function which maps [0,1] to (inf, 0]
    ssim_loss = torch.tan(math.pi / 2 * (1 - ssim_value))

    lnl1_loss = -lnl1_metric(prediction, target)

    return psnr_loss + 20 * ssim_loss + lnl1_loss

train_input_root_dir = 'data/DIV2K_train_HR/train_in'
train_output_root_dir = 'data/DIV2K_train_HR/train_out'
test_input_root_dir = 'data/DIV2K_valid_HR/test_in'
test_output_root_dir = 'data/DIV2K_valid_HR/test_out'

# Define hyperparameters
batch_size = 50
learning_rate = 0.0005
epochs = 200

# Initialize the generator and discriminator
generator = EfficientNetGenerator()
discriminator = Discriminator()

# Define loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = ImageEnhancementDataset(train_input_root_dir, train_output_root_dir, transform=transform, train=True)
test_dataset = ImageEnhancementDataset(test_input_root_dir, test_output_root_dir, transform=transform, train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

vif_metric = VisualInformationFidelity()
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1)
psnr_metric = PeakSignalNoiseRatio()

generated_images_dir = 'generated_images'
shutil.rmtree(generated_images_dir, ignore_errors=True)
os.makedirs(generated_images_dir, exist_ok=True)

# Define the path to the checkpoint directory
checkpoint_dir = 'saved_models/gan/'
os.makedirs(checkpoint_dir, exist_ok=True)

# Check if the checkpoint directory exists
if os.path.exists(checkpoint_dir):
    # List all checkpoint files in the directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if re.match(r'gan_checkpoint_epoch\d+\.pth', f)]
    
    if checkpoint_files:
        # Extract and sort the epoch numbers
        epoch_numbers = [int(re.search(r'gan_checkpoint_epoch(\d+)\.pth', f).group(1)) for f in checkpoint_files]
        epoch_numbers.sort()
        
        # Load the latest checkpoint
        latest_epoch = epoch_numbers[-1]
        latest_checkpoint = f'gan_checkpoint_epoch{latest_epoch:04d}.pth'
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        
        # Load the generator and discriminator models and their states
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        loss_discriminator = checkpoint['loss_discriminator']
        loss_generator = checkpoint['loss_generator']

        # Make sure to set the mode for generator and discriminator
        generator.train()
        discriminator.train()

        print(f"Loaded checkpoint from epoch {epoch}. Resuming training...")
    else:
        epoch = 0
        print("No checkpoint found. Starting training from epoch 1...")
else:
    epoch = 0
    print("Checkpoint directory not found. Starting training from epoch 1...")


best_ssim = 0

# Training loop
for epoch in range(epoch, epochs):
    running_psnr = 0.0
    running_ssim = 0.0
    running_vif = 0.0
    running_lnl1 = 0.0

    # Wrap train_loader with tqdm for progress bar
    for batch_idx, (real_images_input, real_images_output) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):

        current_batch_size = real_images_input.shape[0]

        # Train Discriminator
        optimizer_D.zero_grad()

        # Generate fake images from the generator
        fake_images = generator(real_images_input)
        if not((batch_idx + 1)%4):
            save_image(fake_images, f'{generated_images_dir}/fake_image_epoch{epoch + 1:04d}_batch{batch_idx + 1:02d}.png')

        # Calculate PSNR for this batch and accumulate
        psnr = psnr_metric(fake_images, real_images_output)
        running_psnr += psnr
    
        # Calculate SSIM for this batch and accumulate
        ssim_value = ssim_metric(fake_images, real_images_output)
        running_ssim += ssim_value
        
        # Calculate VIF for this batch and accumulate
        vif_value = vif_metric(fake_images, real_images_output)
        running_vif += vif_value

        # Calculate LNL1 for this batch and accumulate
        lnl1_value = lnl1_metric(fake_images, real_images_output)
        running_lnl1 += lnl1_value

        # Calculate the loss for real and fake images
        real_labels = torch.ones(current_batch_size, 1)
        fake_labels = torch.zeros(current_batch_size, 1)
        
        output_real = discriminator(real_images_output)
        loss_real = criterion(output_real, real_labels)
        
        output_fake = discriminator(fake_images.detach())
        loss_fake = criterion(output_fake, fake_labels)
        
        loss_discriminator = loss_real + loss_fake
        loss_discriminator.backward()
        optimizer_D.step()
        
        # Train Generator
        optimizer_G.zero_grad()
        
        output_fake = discriminator(fake_images)
        bce_loss = criterion(output_fake, real_labels)  # BCELoss for generator
        
        # Calculating custom loss
        custom_loss = PSNR_SSIM_LNL1_loss(fake_images, real_images_output)
        
        # Combine BCELoss and CustomLoss for the generator
        loss_generator = bce_loss + custom_loss
        
        loss_generator.backward()
        optimizer_G.step()
        
    
    average_psnr = running_psnr / len(train_loader)
    average_ssim = running_ssim / len(train_loader)
    average_vif = running_vif / len(train_loader)
    average_lnl1 = running_lnl1 / len(train_loader)

    print(f"Epoch [{epoch + 1}/{epochs}] Loss D: {loss_discriminator.item():.4f} Loss G: {loss_generator.item():.4f}")
    print(f'Epoch [{epoch + 1}/{epochs}] Metrics: PSNR: {average_psnr:.4f} SSIM: {average_ssim:.4f} VIF: {average_vif:.4f} LNL1: {average_lnl1:.4f}')

    # Check the condition and save models if met
    if not((epoch + 1) % 5):
        save_path = f'saved_models/gan/gan_checkpoint_epoch{epoch + 1:04d}.pth'
        torch.save({
            'epoch': epoch + 1,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'loss_discriminator': loss_discriminator.item(),
            'loss_generator': loss_generator.item()
        }, save_path)
        print('Model is saved.')

    if best_ssim < average_ssim:
        best_ssim = average_ssim
        save_path = 'saved_models/gan/best_gan_checkpoint.pth'
        torch.save({
            'generator_state_dict': generator.state_dict()
        }, save_path)
        print('$$$ Best model is saved according to SSIM. $$$')

# Testing loop with tqdm
gan_images_dir = 'results/gan'
os.makedirs(gan_images_dir, exist_ok=True)

running_psnr = 0.0
running_ssim = 0.0
running_vif = 0.0
running_lnl1 = 0.0

best_checkpoint_path = 'saved_models/gan/best_gan_checkpoint.pth'
checkpoint = torch.load(best_checkpoint_path)
generator.load_state_dict(checkpoint['generator_state_dict'])

generator.eval()
with torch.no_grad():
    for batch_idx, (real_image_input, real_image_output) in enumerate(tqdm(test_loader, desc="Testing")):

        # Generate fake images from the generator
        fake_image = generator(real_image_input)
        
        # Calculate PSNR for this batch and accumulate
        psnr = psnr_metric(fake_image, real_image_output)
        running_psnr += psnr
    
        # Calculate SSIM for this batch and accumulate
        
        ssim_value = ssim_metric(fake_image, real_image_output)
        running_ssim += ssim_value
        
        # Calculate VIF for this batch and accumulate
        vif_value = vif_metric(fake_image, real_image_output)
        running_vif += vif_value

        # Calculate LNL1 for this batch and accumulate
        lnl1_value = lnl1_metric(fake_image, real_image_output)
        running_lnl1 += lnl1_value

        # You can save or visualize the generated images as needed
        fake_image = transforms.ToPILImage()(fake_image.squeeze().cpu())
        fake_image.save(os.path.join(gan_images_dir, f"generated_{batch_idx + 1:04d}.png"))

    average_psnr = running_psnr / len(test_loader)
    average_ssim = running_ssim / len(test_loader)
    average_vif = running_vif / len(test_loader)
    average_lnl1 = running_lnl1 / len(test_loader)

    print(f"Mean PSNR between enhanced images and real ones: {average_psnr:.4f}")
    print(f"Mean SSIM between enhanced images and real ones: {average_ssim:.4f}")
    print(f"Mean VIF between enhanced images and real ones: {average_vif:.4f}")
    print(f"Mean LNL1 between enhanced images and real ones: {average_lnl1:.4f}")

