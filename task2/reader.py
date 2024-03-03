import os
import cv2
import numpy as np
import torch

# train_path = "train/"
# val_path = "val/"
# test_path = "test/"

def read_data(train_path, val_path, test_path, PATH):

    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []

    for f in os.listdir(PATH + train_path + "salt/"):
        img = cv2.imread(PATH + train_path + "salt/" + f)
        train_x.append(img)
        train_y.append(np.array([0]))

    for f in os.listdir(PATH + train_path + "silv/"):
        img = cv2.imread(PATH + train_path + "silv/" + f)
        train_x.append(img)
        train_y.append(np.array([1]))

    for f in os.listdir(PATH + val_path + "salt/"):
        img = cv2.imread(PATH + val_path + "salt/" + f)
        val_x.append(img)
        val_y.append(np.array([0]))

    for f in os.listdir(PATH + val_path + "silv/"):
        img = cv2.imread(PATH + val_path + "silv/" + f)
        val_x.append(img)
        val_y.append(np.array([1]))

    for f in os.listdir(PATH + test_path + "salt/"):
        img = cv2.imread(PATH + test_path + "salt/" + f)
        test_x.append(img)
        test_y.append(np.array([0]))

    for f in os.listdir(PATH + test_path + "silv/"):
        img = cv2.imread(PATH + test_path + "silv/" + f)
        test_x.append(img)
        test_y.append(np.array([1]))

    # transfrom_to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    train_x =  torch.tensor(np.array(train_x), dtype=torch.float).permute(0, 3, 1, 2)
    train_y = torch.tensor(np.array(train_y), dtype=torch.float)
    val_x = torch.tensor(np.array(val_x), dtype=torch.float).permute(0, 3, 1, 2)
    val_y = torch.tensor(np.array(val_y), dtype=torch.float)
    test_x = torch.tensor(np.array(test_x), dtype=torch.float).permute(0, 3, 1, 2)
    test_y = torch.tensor(np.array(test_y), dtype=torch.float)

    return train_x, train_y, val_x, val_y, test_x, test_y