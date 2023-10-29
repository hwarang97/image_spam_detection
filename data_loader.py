import os
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

def split_dataset(folder_path, label, val_size=0.1, test_size=0.1):
    all_images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                  if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
    # split dataset
    train_data, temp_data = train_test_split(all_images, test_size=val_size+test_size, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=test_size / (val_size + test_size), random_state=42)

    # assign label
    train_data = [(path, label) for path in train_data]
    val_data = [(path, label) for path in val_data]
    test_data = [(path, label) for path in test_data]

    return train_data, val_data, test_data

# split data
# spam_train, spam_val, spam_test = split_dataset(spam_folder, label=1)
# ham_train, ham_val, ham_test = split_dataset(ham_folder, label=0)

# test code
# print(f' spam train len: {len(spam_train)}')
# print(f' spam val len: {len(spam_val)}')
# print(f' spam test len: {len(spam_test)}')
# print(f' ham train len: {len(ham_train)}')
# print(f' ham val len: {len(ham_val)}')
# print(f' ham test len: {len(ham_test)}')

# combine spam and ham
# train_data = spam_train + ham_train
# val_data = spam_val + ham_val
# test_data = spam_test + ham_test

# testcode
# print(f' train len: {len(train_data)}')
# print(f' val len: {len(val_data)}')
# print(f' test len: {len(test_data)}')
# for item in val_data:
    # print(item[0], item[1])

# shuffle data
# np.random.shuffle(train_data)
# np.random.shuffle(val_data)
# np.random.shuffle(test_data)

class ImageDataset(Dataset):
    def __init__(self, data_list, transform=None):

        '''
        Args:
        - data_list (list of tuples): List of (image_path, label)
        - transform (callable, optional): Optional transform to be applied on an image.
        '''
        
        # filter problematic image
        self.data_list = [item for item in data_list if self._is_valid_image(item[0])]
        self.transform = transform

    def _is_valid_image(self, img_path):
        try:
            with Image.open(img_path) as image:
                image = np.array(image)
            return True
        except:
            return False

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        # img_path = os.path.join(self.image_dir, self.image_list[idx])
        # img_path = os.path.join(self.image_dir, '43A4F70B.1010101@seas.upenn.edu_unchecked.png')

        # open image with PIL
        with Image.open(img_path) as image:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = np.array(image)

        # if grayscale, convert to RGB
        if len(image.shape)==2:
            image = np.stack([image, image, image], axis=-1)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label
    
def get_loaders(train_data, val_data, test_data, batch_size=32):
    transform = A.Compose([
        A.Resize(76, 76),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_dataset = ImageDataset(train_data, transform=transform)
    val_dataset = ImageDataset(val_data, transform=transform)
    test_dataset = ImageDataset(test_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

# train_loader, val_loader, test_loader = get_loaders(train_data, val_data, test_data)

# if __name__ == '__main__':
#     loader = get_loaders('"/mnt/c/Users/Kim Seok Je/Desktop/대학원/데이터보안과 프라이버시/report/personal_image_ham/personal_image_ham"')
#     for images in loader:
#         print(images.shape)

# test code
# image_dir = "/mnt/c/Users/Kim Seok Je/Desktop/대학원/데이터보안과 프라이버시/report/personal_image_ham/personal_image_ham"
# dataset = ImageDataset(image_dir)
# print(len(dataset))

# loader = get_loaders(image_dir)
# for images in loader:
#     print(images.shape)

# Iterate through the train loader and print the shape of images and labels
# for i, (images, labels) in enumerate(train_loader):
#     print(f"Batch {i+1} - Images Shape: {images.shape}, Labels Shape: {labels.shape}")
    
#     # Optional: If you want to stop after a few batches, you can break the loop
#     if i == 2:  # stop after 3 batches
#         break
