import torch
import urllib
from PIL import Image
from torchvision import transforms
import sys
from torchsummary import summary as summary_
import torchvision.models as models
import torch.nn as nn
import numpy as np
import torch.nn.functional
import torchvision

#torch.set_printoptions(precision=30) # 소수점 n째자리까지만 계산
#np.set_printoptions(precision=30) # 소수점 n째자리까지만 계산
torch.set_printoptions(threshold=10000000)   # 행렬의 모든 값 출력

#model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
model = models.mobilenet_v2(pretrained=True)
model.eval()

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256, interpolation = Image.BILINEAR), # bilinear mode : 92.848% / nearest mode : 84.169%
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
#print(output[0])

# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
#print(probabilities)

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)

for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
'''
sys.stdout = open('MobileNetV2_hook_output.txt', 'w') # 파일 쓰기모드 실행

def printnorm(self, input, output):
    print(output)        # 각 layer의 output 출력
    print(output.shape)
    print('-------------------------------------------------------')

for layer in model.modules():
    layer.register_forward_hook(printnorm)

input_tensor = input_tensor.reshape(1, 3, 224, 224) # 내가 짠 코드에서는 (3, 1, 224, 224)의 크기임
out = model(input_tensor)

sys.stdout.close() # 출력 값들을 txt 파일에 저장
'''