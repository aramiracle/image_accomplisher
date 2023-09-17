from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import torch

# Define a custom dataset class for image enhancement tasks.
class ImageEnhancementDataset(Dataset):
    def __init__(self, input_root_dir, output_root_dir, transform=None, train=True):
        """
        Initialize the ImageEnhancementDataset.

        Args:
            input_root_dir (str): Path to the directory containing input images.
            output_root_dir (str): Path to the directory containing output (target) images.
            transform (callable, optional): Optional image transformations to apply.
            train (bool, optional): Set to True for training data; applies data augmentation.
        """
        self.input_root_dir = input_root_dir
        self.output_root_dir = output_root_dir
        self.transform = transform
        self.train = train

        # List all input and output image file paths
        self.input_image_paths = [os.path.join(input_root_dir, fname) for fname in sorted(os.listdir(input_root_dir))]
        self.output_image_paths = [os.path.join(output_root_dir, fname) for fname in sorted(os.listdir(output_root_dir))]

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.input_image_paths)

    def apply_data_augmentation(self, input_image, output_image):
        """
        Apply data augmentation to input and output images.

        Args:
            input_image (PIL.Image): Input image.
            output_image (PIL.Image): Output (target) image.

        Returns:
            PIL.Image: Augmented input image.
            PIL.Image: Augmented output (target) image.
        """
        input_image = self.transform(input_image)
        output_image = self.transform(output_image)

        if self.train:

            # Color Jitter
            color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            input_image = torch.where(input_image == 0 , input_image, color_jitter(input_image))
            output_image = torch.where(output_image == 0, input_image, color_jitter(output_image))

        return input_image, output_image

    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            PIL.Image: Augmented input image.
            PIL.Image: Augmented output (target) image.
        """
        input_img_path = self.input_image_paths[idx]
        output_img_path = self.output_image_paths[idx]

        input_image = Image.open(input_img_path).convert('RGB')
        output_image = Image.open(output_img_path).convert('RGB')

        input_image, output_image = self.apply_data_augmentation(input_image, output_image)

        return input_image, output_image

def get_data_loaders(train_input_root_dir, train_output_root_dir, test_input_root_dir, test_output_root_dir, batch_size):
    """
    Create data loaders for training and testing datasets.

    Args:
        train_input_root_dir (str): Path to the directory containing training input images.
        train_output_root_dir (str): Path to the directory containing training output images.
        test_input_root_dir (str): Path to the directory containing testing input images.
        test_output_root_dir (str): Path to the directory containing testing output images.
        batch_size (int): Batch size for data loaders.

    Returns:
        DataLoader: Training data loader.
        DataLoader: Testing data loader.
    """
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = ImageEnhancementDataset(train_input_root_dir, train_output_root_dir, transform=transform, train=True)
    test_dataset = ImageEnhancementDataset(test_input_root_dir, test_output_root_dir, transform=transform, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader
