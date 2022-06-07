import os
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


class SAIT_dataset(Dataset):
    def __init__(self, path, mode):
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.PILToTensor()
        ])

        self.inputs = glob(os.path.join(path, "SEM", "*"))  # SEM 이미지
        self.inputs = sorted(list(set([i[:-9] for i in self.inputs])))  # iter0, iter1이런거 제거하고 앞에만 남김

        if self.mode != "Test":
            self.gt = sorted(glob(os.path.join(path, "Depth", "*")))  # Depth 이미지

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = [self.transform(Image.open(f"{self.inputs[idx]}_itr{i}.png")) for i in range(4)]  # 파일명 끝에 iter0, iter1 같은거 붙여줌
        if self.mode != "Test":
            gt = self.transform(Image.open(self.gt[idx]))
            return inputs, gt
        return inputs


if __name__ == "__main__":
    dataset = SAIT_dataset("./dataset/Train", "Train")
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    imgs, gt = next(iter(train_dataloader))
    print(f"SEM batch shape: {imgs[0].size()}")  # bs, 1, 66, 45
    print(f"gt batch shape: {gt.size()}")
    # img = train_features[0].squeeze()
    # label = train_labels[0]
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # print(f"Label: {label}")
