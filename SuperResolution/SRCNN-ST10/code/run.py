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
from typing import Tuple, List, Dict, Set, Union, Any, Optional, Callable

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from model import SRCNN

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

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

def train(args, device, model, criterion, optimizer, train_loader) -> None:
    check_point_dir = args.output_dir

    ## Log file for train loss
    train_loss_log_file = os.path.join(check_point_dir, "train_loss_log_{}.csv".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    if not os.path.exists(train_loss_log_file):
        with open(train_loss_log_file, 'w') as f:
            f.write("\tepoch\ttrain_loss\ttrain_acc\n")

    ## Log file for eval loss
    eval_loss_log_file = os.path.join(check_point_dir, "eval_loss_log_{}.csv".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    if not os.path.exists(eval_loss_log_file):
        with open(eval_loss_log_file, 'w') as f:
            f.write("\tepoch\teval_loss\teval_acc\n")


    logger.info(f"********* Training Start *********")
    logger.info(f"\tCheck point directory: {check_point_dir}")
    logger.info(f"\tEpochs: {args.epochs}")
    logger.info(f"\tBatch size: {args.batch_size}")
    
    for num, epoch in enumerate(tqdm(range(args.epochs+1))):
        model.train()

        train_loss = 0.0
        running_loss = 0.0
        running_corrects = 0.0
        for idx, (train_inputs, train_target) in enumerate(train_loader):
            inputs, target = train_inputs.to(device), train_target.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs ,target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects += torch.sum(outputs == target.data)
            if args.do_eval:
                eval_loss, eval_acc = eval(args, device, model, criterion)
                with open(eval_loss_log_file, 'a') as f:
                    f.write(f"{idx}\t{epoch}\t{eval_loss}\t{eval_acc}\n")

        ## Calculate loss & accuracy
        train_loss = running_loss / len(train_loader)
        train_acc = running_corrects.double() / len(train_loader)
        with open(train_loss_log_file, 'a') as f:
            f.write(f"{num}\t{epoch}\t{train_loss/len(train_loader)}\t{train_acc}\n")

        ## Save check point
        check_point = model.state_dict().copy()
        check_point_name = f"check-point_{epoch}.pth"
        check_point_path = os.path.join(check_point_dir, check_point_name)
        torch.save(check_point, check_point_path)
    return check_point

def eval(args, device, model, criterion) -> Tuple[float, float]:
    ## Load evaluation datasets
    with open(args.eval_data_file, mode="rb") as f:
        eval_dataset = pickle.load(f)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    
    eval_loss = 0.0
    running_loss = 0.0
    running_corrects = 0.0
    model.eval()
    for eval_inputs, eval_target in tqdm(eval_loader):
        inputs, target = eval_inputs.to(device), eval_target.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, target)
        running_loss += loss.item()
        
    eval_loss = running_loss / len(eval_loader)
    eval_acc = running_corrects / len(eval_loader)
    return eval_loss, eval_acc

def test(args, device, model, criterion) -> None:
    logger.info(f"Loading test datasets...")
    with open(args.test_data_file, mode="rb") as f:
        test_dataset = pickle.load(f)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    check_point_dir = args.output_dir
    ## Log file for test loss
    test_loss_log_file = os.path.join(check_point_dir, "test_loss_log_{}.csv".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    if not os.path.exists(test_loss_log_file):
        with open(test_loss_log_file, 'w') as f:
            f.write("\ttest_loss\ttest_acc\n")

    test_loss = 0.0
    running_loss = 0.0
    running_corrects = 0.0
    for test_inputs, test_target in tqdm(test_loader):
        inputs, target = test_inputs.to(device), test_target.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, target)
        running_loss += loss.item()
        running_corrects += torch.sum(outputs == target.data)

    ## Calculate loss & accuracy
    test_loss = running_loss / len(test_loader)
    test_loss = running_corrects / len(test_loader)
    with open(test_loss_log_file, 'a') as f:
        f.write(f"0\t{test_loss/len(test_loader)}\t{test_loss}\n")

    return None


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)

    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--out_channels", type=int, default=3)

    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--base_dir", type=str, default="./saved_models")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--train_data_file", type=str)
    parser.add_argument("--eval_data_file", type=str)
    parser.add_argument("--test_data_file", type=str)

    args = parser.parse_args()


    ## Device & Model settings...
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SRCNN(args.in_channels, args.out_channels).to(device)
    criterion = nn.MSELoss()

    if not args.model_name_or_path:
        args.output_dir = os.path.join(args.base_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, mode=0o777, exist_ok=True)
    else:
        model.load_state_dict(torch.load(args.model_name_or_path))
        args.output_dir = args.model_name_or_path

    if args.do_train:
        optimizer = optim.Adam(model.parameters())

        logger.info(f"Loading train datasets...")
        with open(args.train_data_file, mode="rb") as f:
            train_dataset = pickle.load(f)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        last_check_point = train(args, device, model, criterion, optimizer, train_loader)

    if args.do_eval:
        logger.info(f"Loading evaluation datasets...")
        check_point_dir = args.output_dir
        ## Log file for eval loss
        eval_loss_log_file = os.path.join(check_point_dir, "eval_loss_log_{}.csv".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        if not os.path.exists(eval_loss_log_file):
            with open(eval_loss_log_file, 'w') as f:
                f.write("\teval_loss\teval_acc\n")
        
        if "last_check_point" in locals():
            eval_loss, eval_acc = eval(args, device, model, criterion)
            with open(eval_loss_log_file, 'a') as f:
                f.write(f"0\t{eval_loss}\t{eval_acc}\n")
        else:
            model_path = args.model_name_or_path
            model.load_state_dict(torch.load(model_path))
            eval_loss, eval_acc = eval(args, device, model, criterion)
            with open(eval_loss_log_file, 'a') as f:
                f.write(f"0\t{eval_loss}\t{eval_acc}\n")

    if args.do_test:
        if "last_check_point" in locals():
            test(args, device, model, criterion)
        else:
            model_path = args.model_name_or_path
            model.load_state_dict(torch.load(model_path))
            test(args, device, model, criterion)

    return None

if __name__ == "__main__":
    main()