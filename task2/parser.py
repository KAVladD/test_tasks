from os import listdir
import numpy as np
import cv2

salt_path = "salt/"
silv_path = "silvinit/"

salt_n = len(listdir(salt_path))
silv_n = len(listdir(silv_path))

salt_dict = {}
silv_dict = {}

salts_dict = {}

for f in listdir(salt_path):

    f_name = list(f.split('_'))
    if not salt_dict.get(f_name[0]):
        salt_dict[f_name[0]] = 0
    salt_dict[f_name[0]] += 1

    if not salts_dict.get(f_name[0]):
        salts_dict[f_name[0]] = [0, 0]
    salts_dict[f_name[0]][0] += 1

for f in listdir(silv_path):

    f_name = list(f.split('_'))
    if not silv_dict.get(f_name[0]):
        silv_dict[f_name[0]] = 0
    silv_dict[f_name[0]] += 1

    if not salts_dict.get(f_name[0]):
        salts_dict[f_name[0]] = [0, 0]
    salts_dict[f_name[0]][1] += 1


print("\nall: ", salt_n + silv_n, ", ", salt_n / silv_n)

names = set(salts_dict.keys())

train = ["DSC00904", "9019-2", "9037", "9040", "906", "DSC00902",
         "901-new", "9038", "911", "912",
         "915", "DSC00901", "DSC00909", "DSC00913", "DSC00919",
         "9005", "901", "9042-1", "9042-2", "914"]
train_ = np.array([0, 0])
val = ["903", "9016", "9034", "9013", "9013-2"]
val_ = np.array([0, 0])
test = ["9036", "909", "910", "9012", "908", "918", "9007", "9004", "9019-1"]
test_ = np.array([0, 0])

for name in train:
    if name in names:
        names.remove(name)
        train_ += np.array(salts_dict[name])
for name in val:
    if name in names:
        names.remove(name)
        val_ += np.array(salts_dict[name])
for name in test:
    if name in names:
        names.remove(name)
        test_ += np.array(salts_dict[name])

print("\ntrain:", train_, " => ", np.sum(train_), train_[0] / train_[1])
print("  val:", val_, " => ", np.sum(val_), val_[0] / val_[1])
print(" test:", test_, " => ", np.sum(test_), test_[0] / test_[1])

for key in salts_dict:

    name = key
    if name in names:
        while len(name) < 8:
            name = ' ' + name
        print(name, salts_dict[key])

re_shape = [128, 128]

counter1 = 0
counter2 = 0
counter3 = 0

for f in listdir(salt_path):

    f_name = list(f.split('_'))

    if f_name[0] in train:
        img = cv2.imread(salt_path+f)
        img = cv2.resize(img, re_shape)
        cv2.imwrite("train/salt/"+str(counter1)+".png", img)
        counter1 += 1

    if f_name[0] in val:
        img = cv2.imread(salt_path+f)
        img = cv2.resize(img, re_shape)
        cv2.imwrite("val/salt/"+str(counter2)+".png", img)
        counter2 += 1

    if f_name[0] in test:
        img = cv2.imread(salt_path+f)
        img = cv2.resize(img, re_shape)
        cv2.imwrite("test/salt/"+str(counter3)+".png", img)
        counter3 += 1

counter1 = 0
counter2 = 0
counter3 = 0

for f in listdir(silv_path):

    f_name = list(f.split('_'))

    if f_name[0] in train:
        img = cv2.imread(silv_path+f)
        img = cv2.resize(img, re_shape)
        cv2.imwrite("train/silv/"+str(counter1)+".png", img)
        counter1 += 1

    if f_name[0] in val:
        img = cv2.imread(silv_path+f)
        img = cv2.resize(img, re_shape)
        cv2.imwrite("val/silv/"+str(counter2)+".png", img)
        counter2 += 1

    if f_name[0] in test:
        img = cv2.imread(silv_path+f)
        img = cv2.resize(img, re_shape)
        cv2.imwrite("test/silv/"+str(counter3)+".png", img)
        counter3 += 1
