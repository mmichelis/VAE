import torch
import os

from PIL import Image
from torchvision import transforms


class ImageData (torch.utils.data.Dataset):
    def __init__(self, data_dir="Data/"):
        self.images = []

        for f in os.listdir(data_dir):
            im = Image.open(os.path.join(data_dir, f))

            self.images.append(im)


    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, idx):
        im = self.images[idx]

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2*x-1)
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        X = preprocess(im)                  # Shape [3, H, W]

        return X
        