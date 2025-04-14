import glob
import os
from typing import Tuple, List

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from src.dataloaders.custom_datasets.random import generate_random_dataset


class RandomDataset(Dataset):

    def __init__(self, path: str = None, img_shape: Tuple[int, int, int] = (3, 32, 32), dataset_size: int = 10,
                 transform=None, label_as_segmap=False, is_image=True, output_as_float=False, save_to_disk=False):
        self.img_shape = img_shape
        self.dataset_size = dataset_size
        self.transform = transform
        self.label_as_segmap = label_as_segmap
        self.is_image = is_image
        self.output_as_float = output_as_float
        self.save_to_disk = save_to_disk
        self.path = path
        if path and os.path.exists(path):
            self.images = glob.glob(os.path.join(path, "*"))[:self.dataset_size]
            print(f"RandomDataset -> Loaded {len(self.images)} images from path: {path}")
        else:
            self.images = self.generate()

        self.size = len(self.images) if self.images else 0

    def __getitem__(self, index: int):
        img = self.images[index]
        output = 0
        if self.output_as_float:
            output = 0.0
        if self.is_image:
            if isinstance(img, str):
                img = Image.open(img)
            if self.transform:
                img = self.transform(img)

            img = T.Resize((self.img_shape[1], self.img_shape[2]))(img)  # Make sure the image is of the correct size
            if self.label_as_segmap:
                return T.ToTensor()(img), T.ToTensor()(img)[0, :].long()
            return T.ToTensor()(img), output
        return img, output

    def __len__(self):
        return self.size

    def generate(self) -> List:
        return generate_random_dataset(img_shape=self.img_shape,
                                       dataset_size=self.dataset_size,
                                       save_to_disk=self.save_to_disk,
                                       apply_transforms=self.is_image,
                                       output_folder_path=self.path)
