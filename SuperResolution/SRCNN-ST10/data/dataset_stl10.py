import os
import sys
import csv
import glob
import gzip
import math
import json
import time
import random
import logging
import argparse
import pickle
from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

class MyDataset(Dataset):
    def __init__(self, dataset, transform_sr):
        self.dataset = dataset
        self.transform  = transform_sr

    def __getitem__(self, index):
        img_ture , _ = self.dataset[index]
        img_D_sample = self.transform(img_ture)
        img_ture = transforms.functional.to_tensor(img_ture)
        return img_D_sample, img_ture

    def __len__(self):
        return len(self.dataset)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_shuffle", action="store_true")
    parser.add_argument("--output_dir", type=str, default=".")

    parser.add_argument("--train_size", type=int, default=0.8)
    parser.add_argument("--eval_size", type=int, default=0.1)
    parser.add_argument("--test_size", type=int, default=0.1)

    args = parser.parse_args()

    dataset = torchvision.datasets.STL10(root=args.output_dir, split="train", download=args.data_shuffle)
    train_dataset, eval_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, 
                                                                                lengths=[args.train_size, args.eval_size, args.test_size])

    transform_sr = transforms.Compose([
        transforms.Resize(48),
        transforms.Resize(96),
        transforms.ToTensor()
    ])

    train_dataset = MyDataset(train_dataset, transform_sr)
    eval_dataset = MyDataset(eval_dataset, transform_sr)
    test_dataset = MyDataset(test_dataset, transform_sr)

    data_type = ["train", "eval", "test"]
    each_dataset = [train_dataset, eval_dataset, test_dataset]
    for t, d in tqdm(zip(data_type, each_dataset)):
        dataset_path = os.path.join(args.output_dir, f"{t}.pkl")
        with open(dataset_path, mode="wb") as f:
            pickle.dump(d, f)

    return None

if __name__ == "__main__":
    main()