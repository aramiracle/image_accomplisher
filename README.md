# Image Accomplisher Project

## Project Overview

The Image Accomplisher Project is a Python-based project for image enhancement using deep learning techniques. It focuses on improving image quality and detail using convolutional neural networks (CNNs). In this project, a random part of the input image is erased, and the goal is to reproduce and enhance the erased portion.

## Dataset

The main dataset used in this project is a combination of images from the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) from ETH Zurich and the [Linnaeus 5 dataset](http://users.isy.liu.se/cvl/datasets/linnaeus5/) for object recognition.

## Project Components

### Preprocessing (`preprocess.py`)

- **Input and Output Directories:** The script defines input and output directories for the training and testing datasets. In this project, they are set as follows:
  - `train_input_folder`: Directory containing high-resolution input images for training.
  - `test_input_folder`: Directory containing high-resolution input images for testing.
  - `train_save_dir`: Directory to save processed training images.
  - `test_save_dir`: Directory to save processed testing images.

- **Image Preprocessing Functions:**
  - `train_preprocess(input_path, save_path)`: Preprocesses a single training image.
  - `test_preprocess(input_path, save_path)`: Preprocesses a single testing image.

- **Parallel Processing:** The script employs multiprocessing to parallelize the preprocessing of images, which can significantly improve efficiency when dealing with a large number of images.

### Data Loading (`dataloader.py`)

- **Custom Dataset Class:** The script defines a custom dataset class called `ImageEnhancementDataset` for image enhancement tasks.
  - It takes input and output root directories as input and allows for optional image transformations.
  - The `train` parameter specifies whether the dataset is for training (applies data augmentation) or testing.
  - The class lists input and output image file paths and implements data augmentation methods.

- **Data Augmentation:**
  - Data augmentation is applied during training to enhance the model's ability to generalize.
  - Color jittering is used to introduce variations in brightness, contrast, saturation, and hue.

- **Data Loaders:** The script provides functions to create data loaders for both training and testing datasets.

### Generator Models (`model.py`)

#### `Generator` Model

- **Architecture:**
  - The `Generator` model is designed for image enhancement tasks.
  - It consists of convolutional and deconvolutional blocks, allowing it to upscale low-resolution images to high-resolution ones.
  - The architecture includes convolution layers, LeakyReLU activation functions, Batch Normalization, and residual blocks.

- **Residual Blocks:**
  - The `ResidualBlock` class defines a residual block used within the generator.
  - Residual connections are added to help the model learn residual features, which can be crucial for image enhancement.

#### `EfficientNetGenerator` Model

- **Architecture:**
  - The `EfficientNetGenerator` model leverages a pre-trained EfficientNet architecture to extract image features.
  - It removes the fully connected layers at the end of the EfficientNet model and adds additional layers for image enhancement.
  - The model includes convolutional and deconvolutional layers.

- **Feature Extraction:**
  - Features are extracted from the input images using the EfficientNet backbone.
  - A fully connected layer is used to further process the features before combining them with the generator's output.

### Utility Functions (`utils.py`)

- **Loss Functions:**
  - The script defines custom loss functions for training the generator model.
  - `lnl1_metric(prediction, target)`: Calculates the Log-Normalized L1 Loss, which is used as a loss metric.
  - `PSNR_SSIM_LNL1_loss(prediction, target)`: Combines PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index), and LNL1 loss for training.

- **Data Loading and Splitting:**
  - `loading_data(train_input_root_dir, train_output_root_dir, test_input_root_dir, test_output_root_dir, batch_size)`: Loads and splits the data into training and testing data loaders.
  - It allows you to specify a ratio for pretraining if needed.

- **Training and Testing Functions:**
  - `common_train(train_loader, generator, optimizer, epochs, generated_images_dir, checkpoint_dir, best_checkpoint_path, is_pretrain=False)`: Common training function for both pretraining and training phases.
  - `train(train_loader, generator, optimizer, epochs, generated_images_dir, checkpoint_dir, best_checkpoint_path)`: Training function.
  - `pretrain(train_loader, generator, optimizer, epochs, generated_images_dir, checkpoint_dir)`: Pretraining function.
  - `test(test_loader, generator, best_checkpoint_path)`: Testing function for evaluating model performance.

## Usage Instructions

The project is designed for image enhancement tasks and can be adapted for various use cases. To use it effectively, follow these steps:

1. **Install Dependencies:** Make sure you have all required dependencies installed, including PyTorch, torchvision, Pillow (PIL), multiprocessing, and tqdm. You can install them using `pip` if needed.

2. **Prepare Your Dataset:** Organize your dataset with specific directory structures for training and testing images.

3. **Configure Main Script (`main.py`):**
   - Define data directories for training input, training output, testing input, and testing output images.
   - Specify hyperparameters such as batch size, learning rate, and the number of training epochs.
   - Choose between the `Generator` or `EfficientNetGenerator` model and configure loss functions and optimizers.

4. **Run the Main Script:** Execute the `main.py` script to start training and testing. This script acts as the central control for the project.

5. **Generated Images and Model Checkpoints:** The project generates enhanced images and saves them in the `generated_images` directory. Model checkpoints, including the best-performing model, are saved in the `saved_models/unet/` directory.

6. **Evaluate Model Performance:** You can assess the trained model's performance using various metrics such as PSNR, SSIM, and LNL1 on the testing dataset.

Please note that you can further customize the project by adjusting the architecture of the generator model, experimenting with different loss functions, and fine-tuning hyperparameters to best suit your specific image enhancement task.

