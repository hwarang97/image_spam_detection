import os
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_list = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(f'Failed to load {img_path}')

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image
    
def get_loaders(image_dir, batch_size=2):
    transform = A.Compose([
        A.Resize(76, 76),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    dataset = ImageDataset(image_dir=image_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader
    
# test code
image_dir = "/mnt/c/Users/Kim Seok Je/Desktop/대학원/데이터보안과 프라이버시/report/personal_image_ham/personal_image_ham"
dataset = ImageDataset(image_dir)
print(len(dataset))

loader = get_loaders(image_dir)
for images in loader:
    print(images.shape)