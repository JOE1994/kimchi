# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import io
import bson                       # this is installed with the pymongo package
import matplotlib.pyplot as plt
from skimage.data import imread   # or, whatever image library you prefer
import multiprocessing as mp      # will come in handy due to the size of the data
import pprint

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output


# Any results you write to the current directory are saved as output.

# Simple data processing

data = bson.decode_file_iter(open('D:/Download/train_example.bson', 'rb'))

prod_to_category = dict()
prod_to_image = dict()
category_list = dict()

iternum = 0

for c, d in enumerate(data):
    product_id = d['_id']
    category_id = d['category_id'] # This won't be in Test data
    prod_to_category[product_id] = category_id

    if(category_list.setdefault(category_id,-1)==-1):
        category_list[category_id] = iternum
        iternum += 1

    imagelist = []
    for e, pic in enumerate(d['imgs']):
        picture = imread(io.BytesIO(pic['picture']),flatten=True)
        imagelist.append(picture)
        # do something with the picture, etc
        # plt.imshow(picture)
        # plt.show()
    prod_to_image[product_id] = imagelist

pprint.pprint(prod_to_category)
pprint.pprint(category_list)

for key, value in prod_to_category.items():
    prod_to_category[key] = category_list[value]

for key, value in prod_to_image.items():
    print(key)
    for image in value:
        pprint.pprint(image)
        # plt.imshow(image)
        # plt.show()
    print(str(key) + ' is in ' + str(prod_to_category[key]))
'''
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 20
batch_size = 100
learning_rate = 0.001

train_dataset = torch.utils.data.TensorDataset(img, labels.view(-1))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer3 = nn.Sequential(
            nn.Linear(14*14*32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.LogSoftmax())

        self.lstm = nn.LSTM(28,300,2, batch_first=True)

    def forward(self, x):
        # h0 = Variable(torch.randn(2, x.size(0), 300).cuda())
        # c0 = Variable(torch.randn(2, x.size(0), 300).cuda())

        # out, _ = self.lstm(x, (h0, c0))
        # out = self.layer3(out[:, -1, :])

        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.layer3(out)

        return out

net = Net()
net.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch tensor to Variable
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))

    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images).cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()

    print('Accuracy of the network on the 10000 test images: %f %%' % (100.0 * correct / total))

# Save the Model
torch.save(net.state_dict(), 'model.pkl')'''
