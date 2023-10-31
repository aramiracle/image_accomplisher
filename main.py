import torch.optim as optim
from model import Generator  # Import the EfficientNetGenerator or Generator class from your model module
from utils import loading_data, pretrain, train, test  # Import necessary functions from your utils module

# Define data directories
train_input_dir = 'data/DIV2K_train_HR/train_in'
train_output_dir = 'data/DIV2K_train_HR/train_out'
test_input_dir = 'data/DIV2K_valid_HR/test_in'
test_output_dir = 'data/DIV2K_valid_HR/test_out'
generated_images_dir = 'generated_images'
checkpoint_dir = 'saved_models/unet/'
best_checkpoint_path = 'saved_models/unet/best_model_checkpoint.pth'
result_images_dir = 'results/unet'

# Define hyperparameters
pretrain_epochs = 4000
batch_size = 32
learning_rate = 1e-3
final_epochs = 5000

# Initialize the generator
generator = Generator()  # Create an instance of the EfficientNetGenerator

# Define loss function and optimizer for the generator
optimizer = optim.Adam(generator.parameters(), lr=learning_rate)  # Use Adam optimizer

# Load and split data into loaders
pretrain_loader, train_loader, test_loader = loading_data(train_input_dir, train_output_dir, test_input_dir, test_output_dir, batch_size, ratio=0.005)

# Pretrain the generator - Useful if you want to start training from scratch
pretrain(pretrain_loader, generator, optimizer, pretrain_epochs, generated_images_dir, checkpoint_dir)

# Train the generator
train(train_loader, generator, optimizer, final_epochs, generated_images_dir, checkpoint_dir, best_checkpoint_path)

# Test the generator
test(test_loader, generator, best_checkpoint_path, result_images_dir)
