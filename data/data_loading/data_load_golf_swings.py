import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR_PATH = os.path.dirname(CURRENT_DIR_PATH)
GOLF_SWINGS_ROOT_PATH = os.path.join(DATA_DIR_PATH, "golf_swings")


class GolfSwingsData(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "golf_swing_sequence"))))

    def __getitem__(self, idx):
        # load images
        img_path = os.path.join(self.root, "golf_swing_sequence", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        return len(self.imgs)


def get_transform():
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


dataset = GolfSwingsData(GOLF_SWINGS_ROOT_PATH, transforms= get_transform())

data_loader_golf_swings = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False)
