# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

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
        picture = imread(io.BytesIO(pic['picture']))
        imagelist.append(picture)
        # do something with the picture, etc
        # plt.imshow(picture)
        # plt.show()
    prod_to_image[product_id] = imagelist

pprint.pprint(prod_to_category)
pprint.pprint(category_list)

for key, value in prod_to_image.items():
    print(key)
    for image in value:
        plt.imshow(image)
        plt.show()
    print(str(key) + ' is in ' + str(prod_to_category[key]))