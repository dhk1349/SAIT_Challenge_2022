from dataset import SAIT_dataset

import torch
from torch.utils.data import DataLoader

train_dataset = SAIT_dataset("./dataset/Train", "Train")
val_dataset = SAIT_dataset("./dataset/Train", "Val")
test_dataset = SAIT_dataset("./dataset/Train", "Test")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


