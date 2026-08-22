import os
import shutil
import random
import csv

# Define the root directories for the original dataset and the new dataset
original_dataset_dir = 'splitonBSTD/bstd/recognition'
new_dataset_dir = 'splitonBSTD/bstd/SceneTextScriptIdentification'
splits = ['train', 'test']
common_languages = ['hindi', 'english']  # We will copy images from these languages

# Function to get the file paths for images in a folder
def get_image_paths(folder):
    return [os.path.join(folder, img) for img in os.listdir(folder) if img.endswith(('.jpg', '.png', '.jpeg'))]

# Function to create the new folder structure, copy images, and write CSV
def create_language_folders(split):
    split_dir = os.path.join(original_dataset_dir, split)
    language_folders = [lang for lang in os.listdir(split_dir) if lang not in common_languages]

    # Create new train/test directory
    new_split_dir = os.path.join(new_dataset_dir, split)
    os.makedirs(new_split_dir, exist_ok=True)

    # Create the CSV file for this split
    csv_filename = os.path.join(new_dataset_dir, f'{split}.csv')
    with open(csv_filename, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Filepath', 'Language'])  # Write the header row

        for lang in language_folders:
            # Create the new folder for the current language
            new_folder = os.path.join(new_split_dir, f'{lang}_')
            os.makedirs(new_folder, exist_ok=True)

            # Get the image paths for the current language, Hindi, and English
            lang_folder = os.path.join(split_dir, lang)
            hindi_folder = os.path.join(split_dir, 'hindi')
            english_folder = os.path.join(split_dir, 'english')

            lang_images = get_image_paths(lang_folder)
            hindi_images = get_image_paths(hindi_folder)
            english_images = get_image_paths(english_folder)

            # Determine the number of images to copy (based on the number of images in the language folder)
            num_images_to_copy = len(lang_images)

            # Randomly select the required number of Hindi and English images
            hindi_sample = random.sample(hindi_images, num_images_to_copy)
            english_sample = random.sample(english_images, num_images_to_copy)

            # Copy the language images, Hindi images, and English images to the new folder
            for img_path in lang_images:
                new_img_path = shutil.copy(img_path, new_folder)
                csvwriter.writerow([new_img_path, lang])  # Log the copied image with its language

            for img_path in hindi_sample:
                new_img_path = shutil.copy(img_path, new_folder)
                csvwriter.writerow([new_img_path, 'hindi'])  # Log the copied Hindi image

            for img_path in english_sample:
                new_img_path = shutil.copy(img_path, new_folder)
                csvwriter.writerow([new_img_path, 'english'])  # Log the copied English image

            print(f'Created folder {new_folder} with {num_images_to_copy} {lang}, {num_images_to_copy} Hindi, and {num_images_to_copy} English images.')

# Process both the train and test folders and create CSVs in the new directory
for split in splits:
    create_language_folders(split)

print('Process completed. New train and test folders along with CSV files created in', new_dataset_dir)
