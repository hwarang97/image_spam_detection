import os
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_list = os.listdir(image_dir)
        self.transform = transform

        # filer out problematic iamge
        self.image_list = [img for img in self.image_list
                           if self._is_valid_image(os.path.join(image_dir, img))]

    def _is_valid_image(self, img_path):
        try:
            with Image.open(img_path) as image:
                image = np.array(image)
            return True
        except:
            return False

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
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
        
        return image
    
def get_loaders(image_dir, batch_size=32):
    transform = A.Compose([
        A.Resize(76, 76),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    dataset = ImageDataset(image_dir=image_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader
    
if __name__ == '__main__':
    loader = get_loaders('"/mnt/c/Users/Kim Seok Je/Desktop/대학원/데이터보안과 프라이버시/report/personal_image_ham/personal_image_ham"')
    for images in loader:
        print(images.shape)

# test code
# image_dir = "/mnt/c/Users/Kim Seok Je/Desktop/대학원/데이터보안과 프라이버시/report/personal_image_ham/personal_image_ham"
# dataset = ImageDataset(image_dir)
# print(len(dataset))

# loader = get_loaders(image_dir)
# for images in loader:
#     print(images.shape)