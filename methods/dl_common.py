from __future__ import print_function, division

import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from methods import base
from helpers.helper_fcts import (
    EWS_Dataset,
    CenterCrop,
    EncodedToTensor,
    Normalize,
    PadInput,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotate,
    RandomVerticalFlip,
    iou,
)


class DLSegmentationMethod(base.BenchmarkMethod):
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        pretrained: bool = False,
        use_augmentation: bool = True,
        epochs: int = 150,
        batchsize: int = 16,
        eval_batchsize: int = 4,
        lr: float = 0.1,
        momentum: float = 0.9,
        n_workers: int = 8,
    ):
        super().__init__()

        self.model = model
        self.model_name = model_name
        self.pretrained = pretrained
        self.use_augmentation = use_augmentation

        self.epochs = epochs
        self.batchsize = batchsize
        self.eval_batchsize = eval_batchsize
        self.lr = lr
        self.momentum = momentum
        self.n_workers = n_workers

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            lr=self.lr,
            momentum=self.momentum,
            params=self.model.parameters(),
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.best_score = 0
        self.checkpoint_path = self._build_checkpoint_path()
        self.history = []

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    def _build_checkpoint_path(self):
        tag = "pretrained" if self.pretrained else "scratch"
        aug_tag = "aug" if self.use_augmentation else "noaug"
        return f"best_{self.model_name}_{tag}_{aug_tag}.pth"

    def preprocess_image(self, image: np.array):
        return image

    def train(self, train_path="data/train", val_path="data/validation"):
        train_paths = glob.glob(train_path + "/*6.png")
        train_dataset = EWS_Dataset(train_paths, self.get_transforms(testval=False))
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batchsize,
            shuffle=True,
            num_workers=self.n_workers,
            pin_memory=True,
        )

        val_paths = glob.glob(val_path + "/*6.png")
        val_dataset = EWS_Dataset(val_paths, self.get_transforms(testval=True))
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=int(self.eval_batchsize),
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=True,
        )

        self.print_run_info()
        self.model.to(self.device)

        scaler = GradScaler()

        for epoch in tqdm(range(self.epochs), desc="Training \t", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
            temp_score = 0
            running_loss = 0.0

            preds_logged_train = torch.zeros((len(train_dataset), 224 * 224))
            y_logged_train = torch.zeros((len(train_dataset), 224 * 224))

            self.model.train()
            self.model.train_size()

            for i, (X, Y) in enumerate(train_dataloader):
                X = {
                    key: X[key].to(self.device, non_blocking=True)
                    for key in ["image", "iso", "fnumber", "exposure"]
                }
                Y = {key: Y[key].to(self.device, non_blocking=True) for key in ["mask"]}

                with autocast():
                    outputs = self.model(X)
                    loss = self.criterion(outputs["mask"], Y["mask"].long().squeeze(1))

                for param in self.model.parameters():
                    param.grad = None

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                running_loss += loss.item()

                n_samples = Y["mask"].shape[0]
                preds = torch.argmax(outputs["mask"], dim=1).reshape((n_samples, -1))
                preds_logged_train[i * self.batchsize: i * self.batchsize + n_samples] = preds.cpu()
                y_logged_train[i * self.batchsize: i * self.batchsize + n_samples] = Y["mask"].reshape(
                    (n_samples, -1)
                ).cpu()

            del loss, outputs, preds

            self.model.eval()
            self.model.eval_size()

            for X, Y in val_dataloader:
                X = {key: X[key].to(self.device, non_blocking=True) for key in ["image"]}
                Y = {key: Y[key].to(self.device, non_blocking=True) for key in ["mask"]}

                with autocast():
                    outputs = self.model(X)

                outputs["mask"] = outputs["mask"][:, :, :350, :350]
                temp_iou = iou("mask", outputs, Y, self.device, 0).cpu()
                temp_score += temp_iou[temp_iou.isfinite()].sum().numpy()

            temp_score /= len(val_dataset)
            epoch_loss = running_loss / max(len(train_dataloader), 1)
            self.history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "val_iou": float(temp_score),
                }
            )

            if temp_score > self.best_score:
                self.best_score = temp_score
                torch.save({"model_state_dict": self.model.state_dict()}, self.checkpoint_path)

            del outputs

        train_metrics = self.calculate_metrics(preds_logged_train, y_logged_train)

        preds_logged_val = torch.zeros((len(val_dataset), 350 * 350))
        y_logged_val = torch.zeros((len(val_dataset), 350 * 350))

        self.load_model(model_name=self.checkpoint_path)
        self.model.eval()
        self.model.eval_size()

        for i, (X, Y) in enumerate(tqdm(val_dataloader, desc="Validation ")):
            X = {key: X[key].to(self.device, non_blocking=True) for key in ["image"]}
            Y = {key: Y[key].to(self.device, non_blocking=True) for key in ["mask"]}

            with autocast():
                outputs = self.model(X)

            outputs["mask"] = outputs["mask"][:, :, :350, :350]

            n_samples = Y["mask"].shape[0]
            preds = torch.argmax(outputs["mask"], dim=1).reshape((n_samples, -1))
            preds_logged_val[i * self.eval_batchsize: i * self.eval_batchsize + n_samples] = preds.cpu()
            y_logged_val[i * self.eval_batchsize: i * self.eval_batchsize + n_samples] = Y["mask"].reshape(
                (n_samples, -1)
            ).cpu()

        val_metrics = self.calculate_metrics(preds_logged_val, y_logged_val)
        return train_metrics, val_metrics

    def test(self, test_path="data/test"):
        test_paths = sorted(glob.glob(test_path + "/*6.png"))
        test_dataset = EWS_Dataset(test_paths, self.get_transforms(testval=True))
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=int(self.eval_batchsize),
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=True,
        )

        self.load_model(model_name=self.checkpoint_path)

        preds_logged_test = torch.zeros((len(test_dataset), 350 * 350))
        y_logged_test = torch.zeros((len(test_dataset), 350 * 350))

        self.model.eval()
        self.model.eval_size()

        for i, (X, Y) in enumerate(tqdm(test_dataloader, desc="Testing ")):
            X = {key: X[key].to(self.device, non_blocking=True) for key in ["image"]}
            Y = {key: Y[key].to(self.device, non_blocking=True) for key in ["mask"]}

            with autocast():
                outputs = self.model(X)

            outputs["mask"] = outputs["mask"][:, :, :350, :350]

            n_samples = Y["mask"].shape[0]
            preds = torch.argmax(outputs["mask"], dim=1).reshape((n_samples, -1))
            preds_logged_test[i * self.eval_batchsize: i * self.eval_batchsize + n_samples] = preds.cpu()
            y_logged_test[i * self.eval_batchsize: i * self.eval_batchsize + n_samples] = Y["mask"].reshape(
                (n_samples, -1)
            ).cpu()

        test_metrics = self.calculate_metrics(preds_logged_test, y_logged_test)
        return test_metrics, preds_logged_test, y_logged_test

    def load_model(self, model_name=None):
        checkpoint = torch.load(model_name or self.checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        return

    def save_history_csv(self, output_path):
        if not self.history:
            return

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(self.history).to_csv(output_path, index=False)

    def print_run_info(self):
        print()
        print("-" * 80)
        print("training for ", self.epochs, " epochs")
        print()
        print("Starting training of the model:", self.model_name)
        print("Training is using:", self.device)
        if torch.cuda.is_available():
            print(torch.cuda.get_device_name(0))
        print("-" * 80)
        print()

    def get_transforms(self, testval: bool):
        transforms = []
        if testval:
            transforms.append(EncodedToTensor())
            transforms.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
            transforms.append(PadInput((352, 352)))
        else:
            transforms.append(EncodedToTensor())
            if self.use_augmentation:
                transforms.append(RandomRotate(20))
                transforms.append(RandomCrop((224, 224)))
                transforms.append(RandomHorizontalFlip())
                transforms.append(RandomVerticalFlip())
            else:
                transforms.append(CenterCrop((224, 224)))
            transforms.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

        return Compose(transforms)


class WrappedSegmentationModel(nn.Module):
    def __init__(self, name, network):
        super().__init__()
        self.name = name
        self.network = network
        self.evaluation = False

    def eval_size(self):
        self.evaluation = True

    def train_size(self):
        self.evaluation = False

    def forward(self, x):
        image = x["image"]
        masks = self.network(image)
        return {"mask": masks}


