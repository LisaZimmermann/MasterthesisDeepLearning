import cv2
import os
import numpy as np
import io
import scipy.ndimage as ndimage
from PIL import Image
#from Alexnet_ACM import Alexnet_ACM
import matplotlib.pyplot as plt
import matplotlib.image as implt
from matplotlib.pyplot import imshow
import matplotlib.colors as colshow
import colorsys

#label_path = "C://Users//Lisa//Pictures//Amsterdam_Juli_2018"
#img_testset = [label[:-5] for label in os.listdir(label_path)]
#feature_images_path = "C://Users//Lisa//acm2017_cnnvisualize//Code//human_features//windowed//"
#print(img_testset)

#Import test image
def rgb2hsv(testimg):
    return testimg.convert('HSV')

def HSVColor(img):
    if isinstance(img,Image.Image):
        r,g,b = img.split()
        Hdat = []
        Sdat = []
        Vdat = []
        for rd,gn,bl in zip(r.getdata(),g.getdata(),b.getdata()) :
            h,s,v = colorsys.rgb_to_hsv(rd/255.,gn/255.,bl/255.)
            Hdat.append(int(h*255.))
            Sdat.append(int(s*255.))
            Vdat.append(int(v*255.))
        r.putdata(Hdat)
        g.putdata(Sdat)
        b.putdata(Vdat)
        return Image.merge('RGB',(r,g,b))
    else:
        return None


img = Image.open("DSC00095.JPG")
#img_converted = HSVColor(img)
#img_converted.save("test.JPG")
#testimg = cv2.COLOR_BGR2HSV(img)
#print(testimg)
img1=implt.imread('DSC00095.JPG')
#print(img1)
#imgplot = plt.imshow(img1)
#plt.show()
img2 = colshow.rgb_to_hsv(img1)
imgplot = plt.imshow(img2)
s_image = img2[:,:,2]
#plt.imshow(s_image)
#plt.show()
#gaussian filter
# Note the 0 sigma for the last axis, we don't wan't to blurr the color planes together!
img_gaussian = ndimage.gaussian_filter(s_image, sigma=(5), order=0)
plt.imshow(img_gaussian, interpolation='nearest')
plt.show()





