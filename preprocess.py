import os
import torchvision
from torchvision.transforms import v2 as transforms
from PIL import Image
import multiprocessing

torchvision.disable_beta_transforms_warning()

# Define the input and output directories
train_input_folder = 'data/DIV2K_train_HR/original'
test_input_folder = 'data/DIV2K_valid_HR/original'
train_save_dir = 'data/DIV2K_train_HR'
test_save_dir = 'data/DIV2K_valid_HR'

def train_preprocess(input_path, save_path):
    os.makedirs(os.path.join(save_path, 'train_in'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'train_out'), exist_ok=True)

    transform_1 = transforms.Resize((64, 64))
    transform_2 = transforms.RandomErasing(p=1,scale=(0.1, 0.3))

    image = Image.open(input_path)

    transformed_image_1 = transform_1(image)
    transformed_image_2 = transform_2(transformed_image_1)

    transformed_image_1.save(os.path.join(save_path, 'train_out', os.path.basename(input_path)))
    transformed_image_2.save(os.path.join(save_path, 'train_in', os.path.basename(input_path)))

def test_preprocess(input_path, save_path):
    os.makedirs(os.path.join(save_path, 'test_in'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'test_out'), exist_ok=True)

    image = Image.open(input_path)
    height, width = image.size
    width, height = (width // 128 * 32, height // 128 * 32 )

    transform_1 = transforms.Resize((width, height))
    transform_2 = transforms.RandomErasing(p=1,scale=(0.1, 0.3))

    transformed_image_1 = transform_1(image)
    transformed_image_2 = transform_2(transformed_image_1)

    transformed_image_1.save(os.path.join(save_path, 'test_out', os.path.basename(input_path)))
    transformed_image_2.save(os.path.join(save_path, 'test_in', os.path.basename(input_path)))

def process_images(input_folder, save_dir, preprocess_func):
    os.makedirs(save_dir, exist_ok=True)

    image_files = os.listdir(input_folder)
    input_paths = [os.path.join(input_folder, file) for file in image_files]

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool.starmap(preprocess_func, zip(input_paths, [save_dir] * len(input_paths)))
    pool.close()
    pool.join()

if __name__ == "__main__":
    # Process the train and test sets using multiprocessing
    process_images(train_input_folder, train_save_dir, train_preprocess)
    process_images(test_input_folder, test_save_dir, test_preprocess)

