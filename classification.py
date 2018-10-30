
import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
import os

alexnet = models.alexnet(pretrained=True)
alexnet.eval()

image_path = "C://Users//Lisa//Documents//bmw_abschlussarbeit//exd_download//bilder handy_1//bilder handy_1//airbag//images"
LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Scale(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
])

label_list = os.listdir(image_path)
for label in label_list:
    img =Image.open(image_path + "//" + label)
    img_tensor = preprocess(img)
    img_tensor.unsqueeze_(0)
    img_variable = Variable(img_tensor)
    fc_out = alexnet(img_variable)

    labels = {int(key):value for (key, value)
              in requests.get(LABELS_URL).json().items()}
    print(labels[fc_out.data.numpy().argmax()])
