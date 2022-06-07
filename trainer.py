import torch
import torch.nn as nn
import cv2
import os


class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, test_loader, output_dir):
        self.model = model
        self.loss = nn.MSELoss()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.optimizer = optimizer

        self.output_dir = output_dir

    def train(self):
        for imgs, gt in iter(self.train_loader):
            imgs = torch.cat(imgs, 1).cuda()
            output = self.model(imgs)
            loss = self.loss(output, gt.cuda())

            print("training loss: {}".format(loss))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def val(self):
        self.model.eval()
        loss_list = []
        for imgs, gt in iter(self.val_loader):
            imgs = torch.cat(imgs, 1).cuda()
            output = self.model(imgs)
            mse = self.loss(output, gt.cuda()).unsqueeze(0)
            loss_list.append(mse)

        rmse = torch.sqrt(torch.sum(torch.cat(loss_list)) / len(self.val_loader))
        print("*************************validation rmse*************************: {}".format(rmse))

        self.model.train()

    def test(self):
        self.model.eval()
        for imgs, img_path in iter(self.test_loader):
            imgs = torch.cat(imgs, 1).cuda()
            output = self.model(imgs)

            output *= 255

            cv2.imwrite("{}/{}.jpg".format(self.output_dir, img_path[0][19:]),
                        output[0].permute(1, 2, 0).cpu().detach().numpy())
