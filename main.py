from dataset import SAIT_dataset
from trainer import Trainer
from model import Network
import os

import torch
from torch.utils.data import DataLoader


def main():
    num_epoch = 50
    load_model = False
    model_dir = "/home/seunghyeon/Desktop/MIPAL/lecture/special_topic_ml/SAIT_Challenge_2022-main/model"
    output_dir = "/home/seunghyeon/Desktop/MIPAL/lecture/special_topic_ml/SAIT_Challenge_2022-main/test_output"

    train_dataset = SAIT_dataset("./dataset/Train", "Train")
    val_dataset = SAIT_dataset("./dataset/Validation", "Val")
    test_dataset = SAIT_dataset("./dataset/Test", "Test")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = Network().cuda()
    if load_model:
        model = torch.load('/home/seunghyeon/Desktop/MIPAL/lecture/special_topic_ml/SAIT_Challenge_2022-main/model/model_49.pth')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

    trainer = Trainer(model, optimizer, train_loader, val_loader, test_loader, output_dir)

    for i in range(num_epoch):
        trainer.train()
        if i != 0:
            if i % 5 == 0:
                trainer.val()
            if i % 10 == 0 or i == num_epoch-1:
                torch.save(model, os.path.join(model_dir, "model_{}.pth".format(i)))

    trainer.test()


if __name__ == "__main__":
    main()