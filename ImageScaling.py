import cv2
import os
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt

label_path = "C://Users//Lisa//Pictures//Amsterdam_Juli_2018//"


img_testset = os.listdir(label_path)
print(img_testset)

newpath = r"C://Users//Lisa//Pictures//Test"
if not os.path.exists(newpath):
    os.makedirs(newpath)

for i in img_testset:
    print(label_path+i)
    img = Image.open(label_path + i)
    img_rotated = img.rotate(5)
    img_rotated.save(label_path + i[:-4] + "_rotated.jpg")