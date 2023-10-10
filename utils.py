import os
import re
import shutil
import torch
import math
from torchvision import transforms
from data_loader import ImageEnhancementDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.utils import save_image
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure, VisualInformationFidelity, PeakSignalNoiseRatio

# Custom loss metric: LNL1 (Log-Normalized L1 Loss)
def lnl1_metric(prediction, target):
    max_value = torch.maximum(prediction, target) + 1e-3 * torch.ones_like(prediction)
    norm1 = torch.absolute(prediction - target)
    normalize_norm1 = torch.mean(torch.mul(norm1, max_value.pow(-1)))
    lnl1_value = -20 * torch.log10(normalize_norm1)

    return lnl1_value

# Combined loss metric including PSNR, SSIM, and LNL1
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

# Function for loading data and creating data loaders
def loading_data(train_input_dir, train_output_dir, test_input_dir, test_output_dir, batch_size, ratio=0.01):
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the entire dataset
    full_train_dataset = ImageEnhancementDataset(train_input_dir, train_output_dir, transform=transform, train=True)

    # Calculate the number of samples for the pretrain and train datasets
    total_samples = len(full_train_dataset)
    pretrain_samples = int(ratio * total_samples)

    # Create random indices for the pretrain and train datasets
    indices = list(range(total_samples))
    pretrain_indices = indices[:pretrain_samples]
    train_indices = indices[pretrain_samples:]

    # Create SubsetRandomSamplers for the pretrain and train datasets
    pretrain_sampler = SubsetRandomSampler(pretrain_indices)
    train_sampler = SubsetRandomSampler(train_indices)

    # Create DataLoader instances for pretrain and train datasets
    pretrain_loader = DataLoader(full_train_dataset, batch_size=batch_size, sampler=pretrain_sampler)
    train_loader = DataLoader(full_train_dataset, batch_size=batch_size, sampler=train_sampler)

    # Load the test dataset
    test_dataset = ImageEnhancementDataset(test_input_dir, test_output_dir, transform=transform, train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return pretrain_loader, train_loader, test_loader

# Common training function
def common_train(train_loader, generator, optimizer, epochs, generated_images_dir, checkpoint_dir, best_checkpoint_path, is_pretrain=False):
    shutil.rmtree(generated_images_dir, ignore_errors=True)
    os.makedirs(generated_images_dir, exist_ok=True)

    os.makedirs(checkpoint_dir, exist_ok=True)

    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if re.match(r'model_checkpoint_epoch\d+\.pth', f)]
        
        if checkpoint_files:
            epoch_numbers = [int(re.search(r'model_checkpoint_epoch(\d+)\.pth', f).group(1)) for f in checkpoint_files]
            epoch_numbers.sort()
            
            epoch = epoch_numbers[-1]
            latest_checkpoint = f'model_checkpoint_epoch{epoch:04d}.pth'
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            
            checkpoint = torch.load(checkpoint_path)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            generator.train()

            print(f"Loaded checkpoint from epoch {epoch}. Resuming training...")
        else:
            epoch = 0
            print("No checkpoint found. Starting training from epoch 1...")
    else:
        epoch = 0
        print("Checkpoint directory not found. Starting training from epoch 1...")

    vif_metric = VisualInformationFidelity()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1)
    psnr_metric = PeakSignalNoiseRatio()

    best_loss = float('inf')

    for epoch in range(epoch, epochs):
        running_psnr = 0.0
        running_ssim = 0.0
        running_vif = 0.0
        running_lnl1 = 0.0
        running_loss = 0.0

        for batch_idx, (real_images_input, real_images_output) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):

            optimizer.zero_grad()
            
            fake_images = generator(real_images_input)

            loss = PSNR_SSIM_LNL1_loss(fake_images, real_images_output)
            running_loss += loss

            loss.backward()
            optimizer.step()

            psnr = psnr_metric(fake_images, real_images_output)
            running_psnr += psnr
        
            ssim_value = ssim_metric(fake_images, real_images_output)
            running_ssim += ssim_value
            
            vif_value = vif_metric(fake_images, real_images_output)
            running_vif += vif_value

            lnl1_value = lnl1_metric(fake_images, real_images_output)
            running_lnl1 += lnl1_value

            average_loss = running_loss / (batch_idx + 1)
            average_psnr = running_psnr / (batch_idx + 1)
            average_ssim = running_ssim / (batch_idx + 1)
            average_vif = running_vif / (batch_idx + 1)
            average_lnl1 = running_lnl1 / (batch_idx + 1)

            if is_pretrain:
                print(f'Epoch [{epoch + 1}/{epochs}] Batch[{batch_idx + 1}/{len(train_loader)}] Loss: {average_loss:.4f} Metrics: PSNR: {average_psnr:.4f} SSIM: {average_ssim:.4f} VIF: {average_vif:.4f} LNL1: {average_lnl1:.4f}')
                save_image(torch.cat((fake_images, real_images_input, real_images_output), dim=0), f'{generated_images_dir}/fake_image_epoch{epoch + 1:04d}_batch{batch_idx + 1:02d}.png', nrow=8)
            elif (batch_idx + 1) % 5 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}] Batch[{batch_idx + 1}/{len(train_loader)}] Loss: {average_loss:.4f} Metrics: PSNR: {average_psnr:.4f} SSIM: {average_ssim:.4f} VIF: {average_vif:.4f} LNL1: {average_lnl1:.4f}')
                save_image(torch.cat((fake_images, real_images_input, real_images_output), dim=0), f'{generated_images_dir}/fake_image_epoch{epoch + 1:04d}_batch{batch_idx + 1:02d}.png', nrow=8)

        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {average_loss:.4f}")
        print(f'Epoch [{epoch + 1}/{epochs}] Metrics: PSNR: {average_psnr:.4f} SSIM: {average_ssim:.4f} VIF: {average_vif:.4f} LNL1: {average_lnl1:.4f}')

        save_path = f'{checkpoint_dir}/model_checkpoint_epoch{epoch + 1:04d}.pth'
        torch.save({
            'generator_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        print('Model is saved.')

        if is_pretrain:
            continue

        if best_loss > average_loss:
            best_loss = average_loss
            torch.save({
                'generator_state_dict': generator.state_dict()
            }, best_checkpoint_path)
            print('$$$ Best model is saved according to loss. $$$')

# Training function
def train(train_loader, generator, optimizer, epochs, generated_images_dir, checkpoint_dir, best_checkpoint_path):
    common_train(train_loader, generator, optimizer, epochs, generated_images_dir, checkpoint_dir, best_checkpoint_path, is_pretrain=False)

# Pretraining function
def pretrain(train_loader, generator, optimizer, epochs, generated_images_dir, checkpoint_dir):
    common_train(train_loader, generator, optimizer, epochs, generated_images_dir, checkpoint_dir, "", is_pretrain=True)

# Testing function
def test(test_loader, generator, best_checkpoint_path, result_images_dir):
    # Testing loop with tqdm
    os.makedirs(result_images_dir, exist_ok=True)

    running_psnr = 0.0
    running_ssim = 0.0
    running_vif = 0.0
    running_lnl1 = 0.0

    checkpoint = torch.load(best_checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])

    vif_metric = VisualInformationFidelity()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1)
    psnr_metric = PeakSignalNoiseRatio()

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
            fake_image.save(os.path.join(result_images_dir, f"generated_{batch_idx + 1:04d}.png"))

        average_psnr = running_psnr / len(test_loader)
        average_ssim = running_ssim / len(test_loader)
        average_vif = running_vif / len(test_loader)
        average_lnl1 = running_lnl1 / len(test_loader)

        print(f"Mean PSNR between enhanced images and real ones: {average_psnr:.4f}")
        print(f"Mean SSIM between enhanced images and real ones: {average_ssim:.4f}")
        print(f"Mean VIF between enhanced images and real ones: {average_vif:.4f}")
        print(f"Mean LNL1 between enhanced images and real ones: {average_lnl1:.4f}")
