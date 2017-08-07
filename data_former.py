import numpy as np
import cv2
import pandas as pd
from collections import Counter
import random

def counter(data):
    res = [0] * 3
    for item in data:
        res[item[1][0]] += 1
    return res

train_data = np.load('train_01.npy') 

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

min_len = min(counter(train_data))

w = []
a = []
d = []
wa = []
wd = []

for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [0]:
        w.append([img,choice])     
    elif choice == [1]:
        wa.append([img,choice])
    elif choice == [2]:
        wd.append([img,choice])

final_data = w[:min_len] + wa[:min_len] + wd[:min_len]
random.shuffle(final_data)

df2 = pd.DataFrame(final_data)
print(Counter(df2[1].apply(str)))

np.save('training_data_f.npy', final_data)