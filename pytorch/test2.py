"""
This code is based on:
-- https://www.kaggle.com/lamdang/fast-shuffle-bson-generator-for-keras
-- https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson

(c) Aleksei Tiulpin, 2017

"""

import os
import pandas as pd
import numpy as np
import bson
import cv2
import random
from tqdm import tqdm
import struct
from PIL import Image
from  torchvision import transforms as transf
import torch.utils.data as data_utils
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


def read_bson(bson_path, with_categories):
    """
    Reads BSON
    """
    offset = 0
    rows = {}
    with open(bson_path, "rb") as f, tqdm(total=7069896) as pbar:
        f.seek(offset)
        records_read = 0
        while True:
            item_length_bytes = f.read(4)
            if len(item_length_bytes) == 0:
                break

            length = struct.unpack("<i", item_length_bytes)[0]
            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length

            item = bson.BSON.decode(item_data)
            product_id = item["_id"]
            num_imgs = len(item["imgs"])

            row = [num_imgs, offset, length]
            if with_categories:
                row += [item["category_id"]]
            rows[product_id] = row

            offset += length
            f.seek(offset)
            records_read += 1
            pbar.update()
        pbar.close()
    columns = ["num_imgs", "offset", "length"]
    if with_categories:
        columns += ["category_id"]

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "product_id"
    df.columns = columns
    df.sort_index(inplace=True)
    return df


def make_category_tables(categories_path):
    """
    Converts category name into an index [0, N-1]
    """
    categories_df = pd.read_csv(categories_path, index_col="category_id")
    categories_df["category_idx"] = pd.Series(range(len(categories_df)), index=categories_df.index)

    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat


def get_obs(fname, offset, length):
    fobj = open(fname, 'rb')
    fobj.seek(offset)
    res = bson.BSON.decode(fobj.read(length))
    fobj.close()
    return res


class CdiscountDataset(data_utils.Dataset):
    def __init__(self, dataset, split, transform):
        self.dataset = dataset
        self.metadata = split
        self.transform = transform

    def __getitem__(self, index):
        entry = self.metadata.iloc[index]
        num_imgs, offset, length, target = entry
        obs = get_obs(self.dataset, offset, length)
        keep = np.random.choice(len(obs['imgs']))
        byte_str = obs['imgs'][keep]['picture']
        img = cv2.imdecode(np.fromstring(byte_str, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = self.transform(img)

        return img, target

    def __len__(self):
        return self.metadata.index.values.shape[0]


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1)


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, pooling=2):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1 + (pooling//3)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(pooling, stride=stride),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels),
        )


    def forward(self, x):
        out = self.layer1(x)
        out += self.layer2(x)
        return out


class ResidualBlock2(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        out = self.layer1(x)
        out += x
        return out


# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer2 = self.residual(64, 128, 3)
        self.layer3 = self.residual(128, 256, 3)
        self.layer4 = self.residual(256, 728, 3)
        self.layer5 = self.residual2(728)
        self.layer6 = self.residual2(728)
        self.layer7 = self.residual2(728)
        self.layer8 = self.residual2(728)
        self.layer9 = self.residual2(728)
        self.layer10 = self.residual2(728)
        self.layer11 = self.residual2(728)
        self.layer12 = self.residual2(728)
        self.layer13 = self.residual(728, 1024, 3)

        self.layer14 = nn.Sequential(
            nn.Conv2d(1024, 1536, kernel_size=3, padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            nn.Conv2d(1536, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.AvgPool2d(6)
        )

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048, 5270),)

        self.test = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Linear(50*50*16, 1024),
            nn.ReLU(),
            nn.Linear(1024,5270),
        )

    def residual(self, ind, outd, pooling):
        layers = []
        layers.append(ResidualBlock(ind, outd, pooling=pooling))
        return nn.Sequential(*layers)

    def residual2(self, d):
        layers = []
        layers.append(ResidualBlock2(d))
        return nn.Sequential(*layers)

    def forward(self, x):
        # h0 = Variable(torch.randn(2, x.size(0), 300).cuda())
        # c0 = Variable(torch.randn(2, x.size(0), 300).cuda())

        # out, _ = self.lstm(x, (h0, c0))
        # out = self.layer3(out[:, -1, :])

        '''out = self.layer1(x)
        out_1 = self.layer2_1(out)
        out_2 = self.layer2_2(out)
        out_3 = self.layer2_3(out)
        out = out_1 + out_2 + out_3
        out = out.view(out.size(0), -1)
        out = self.layer3(out)'''

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = self.layer14(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


net = Net()
net.cuda()

learning_rate=0.001

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-5)

if __name__ == "__main__":
    TRAIN_BSON_FILE = 'C:/test/train.bson'
    TEST_BSON_FILE = 'C:/test/test.bson'
    CATEGS = 'D:/Download/category_names.csv'
    # Chenge this weh running on your machine
    N_TRAIN = 7069896
    R_TRAIN = 256
    R_TEST = 1
    BS = 256
    N_THREADS = 1
    EPOCH = 100

    # mapping the catigores into 0-5269 range
    cat2idx, idx2cat = make_category_tables(CATEGS)

    # Dataset and loader

    # Scanning the metadata
    meta_data = read_bson(TRAIN_BSON_FILE, with_categories=True)
    meta_data.category_id = np.array([cat2idx[ind] for ind in meta_data.category_id])

    temp = np.arange(N_TRAIN)
    np.random.shuffle(temp)

    train_sample = temp[0:7000000]
    test_sample = temp[7000000:7069896]

    train_data = meta_data.iloc[train_sample]
    test_data = meta_data.iloc[test_sample]

    print(train_data)

    training_category = np.unique(np.array(train_data.category_id))
    test_category = np.unique(np.array(test_data.category_id))

    both = np.intersect1d(training_category, test_category)

    both_ratio = 1.0 * len(both)/len(test_category)

    print(both_ratio)

    train_dataset = CdiscountDataset(TRAIN_BSON_FILE, train_data, transf.Compose([
        transf.Scale(200),
        transf.RandomCrop(180),
        transf.RandomHorizontalFlip(),
        transf.ToTensor(),
        transf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))
    loader = data_utils.DataLoader(train_dataset, batch_size=BS, shuffle=True)

    test_dataset = CdiscountDataset(TRAIN_BSON_FILE, test_data, transf.Compose([
        transf.Scale(200),
        transf.CenterCrop(180),
        transf.ToTensor(),
        transf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))
    test_loader = data_utils.DataLoader(test_dataset, batch_size=1, shuffle=False)
    # Let's go fetch some data!

    for epoch in range(EPOCH):
        pbar = tqdm(total=len(loader))
        for i, (batch, target) in enumerate(loader):
            # Convert torch tensor to Variable
            images = Variable(batch.cuda())
            labels = Variable(target.cuda())

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            pbar.update()
        pbar.close()

        # Test the Model
        net.eval()
        correct = 0
        total = 0
        for batch, target in test_loader:
            images = Variable(batch).cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted.cpu() == target).sum()
        net.train()
        print('%dth accuracy of the network on the %d test images: %f %%' % (epoch, len(test_loader), (100.0 * correct / total)))

    # Save the Model
    torch.save(net.state_dict(), 'model.pkl')

