import cv2
import os
import numpy as np
import io
from PIL import Image
from Alexnet_ACM import Alexnet_ACM
import matplotlib.pyplot as plt

label_path = "C://Users//Lisa//Pictures//Amsterdam_Juli_2018"
img_testset = [label[:-5] for label in os.listdir(label_path)]
feature_images_path = "C://Users//Lisa//acm2017_cnnvisualize//Code//human_features//windowed//"
print(img_testset)