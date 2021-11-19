# train_hook에서 저장한 weight 및 bias를 불러오는 코드
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transfroms
import time

start = time.time() # 시간 측정 시작

class ConvNet(nn.Module):
    def __init__(self):  # layer 정의
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512,10)


    def forward(self, input):
        x = self.pool1(F.relu(self.conv1(input)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #x = F.softmax(self.fc2(x))
        return x

model = ConvNet()
model.load_state_dict(torch.load('save_weight.pt')) # train_hook 에서 저장한 weight 및 bias 불러오기

train_set = torchvision.datasets.MNIST(
    root='./data/MNIST',
    train=True,
    download=True,
    transform=transfroms.Compose([
        transfroms.ToTensor()  # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    ])
)

examples = enumerate(train_set)
batch_idx, (example_data, example_targets) = next(examples)
example_data = example_data.reshape(1, 1, 28, 28)
input = torch.Tensor(example_data)

print(model(input))
print(F.softmax(model(input)))
print("걸리는 시간 :", time.time() - start) # 걸리는 시간 출력


'''
# model 전체를 불러오려고 했으나 실패한 코드
import torch
import torchvision
import torchvision.transforms as transfroms
import train_hook

weight = torch.load('save_weight.pt')
#print(torch.load('save_weight.pt'))

model = torch.load('save_model.pt')


print(model.state_dict())
#model.load_state_dict(weight)
# model.eval()

train_set = torchvision.datasets.MNIST(
    root='./data/MNIST',
    train=True,
    download=True,
    transform=transfroms.Compose([
    transfroms.ToTensor()  # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    ])
)

examples = enumerate(train_set)
batch_idx, (example_data, example_targets) = next(examples)
example_data = example_data.reshape(1, 1, 28, 28)
input = torch.Tensor(example_data)
#print(example_data)
print(model(input))
'''