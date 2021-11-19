import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transfroms
import sys
import time

sys.stdout = open('output_and_bias.txt', 'w') # 파일 쓰기모드 실행

torch.set_printoptions(threshold=10000000)   # 행렬의 모든 값 출력

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

print(device + " is available")

learning_rate = 0.001
batch_size = 50
num_classes = 10
epochs = 5

# MNIST 데이터셋 로드
train_set = torchvision.datasets.MNIST(
    root='./data/MNIST',
    train=True,
    download=True,
    transform=transfroms.Compose([
        transfroms.ToTensor()  # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    ])
)

test_set = torchvision.datasets.MNIST(
    root='./data/MNIST',
    train=False,
    download=True,
    transform=transfroms.Compose([
        transfroms.ToTensor()  # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    ])
)

# train_loader, test_loader 생성
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

# input size를 알기 위해서
examples = enumerate(train_set)
batch_idx, (example_data, example_targets) = next(examples)
example_data.shape

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


model = ConvNet().to(device)  # CNN instance 생성
# Cost Function과 Optimizer 선택
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# train
for epoch in range(epochs):  # epochs수만큼 반복
    avg_cost = 0

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()  # 모든 model의 gradient 값을 0으로 설정
        hypothesis = model(data)  # 모델을 forward pass해 결과값 저장
        cost = criterion(hypothesis, target)  # output과 target의 loss 계산
        cost.backward()  # backward 함수를 호출해 gradient 계산
        optimizer.step()  # 모델의 학습 파라미터 갱신
        avg_cost += cost / len(train_loader)  # loss 값을 변수에 누적하고 train_loader의 개수로 나눔 = 평균
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))


# test
model.eval()  # evaluate mode로 전환 dropout 이나 batch_normalization 해제
with torch.no_grad():  # grad 해제
    correct = 0
    total = 0

    for data, target in test_loader: # 1만개 데이터셋
        data = data.to(device)
        target = target.to(device)
        out = model(data)
        preds = torch.max(out.data, 1)[1]  # 출력이 분류 각각에 대한 값으로 나타나기 때문에, 가장 높은 값을 갖는 인덱스를 추출
        total += len(target)  # 전체 클래스 개수
        correct += (preds == target).sum().item()  # 예측값과 실제값이 같은지 비교

    print('Test Accuracy: ', 100. * correct / total, '%')

#torch.save(model, 'train_hook.pt') # 모델 저장

def printnorm(self, input, output):
    print(output)        # 각 layer의 output 출력
    print(output.shape)

# forward_hook 등록
model.conv1.register_forward_hook(printnorm)
model.pool1.register_forward_hook(printnorm)
model.conv2.register_forward_hook(printnorm)
model.pool2.register_forward_hook(printnorm)
model.fc1.register_forward_hook(printnorm)
model.fc2.register_forward_hook(printnorm)

start = time.time() # 시간측정 시작

example_data = example_data.reshape(1,1,28,28)
#print(example_data)
input = torch.Tensor(example_data)
# 첫번째 인풋인 5에 대한 연산 결과(그리고 softmax 함수에 대입한 결과)
print('----------------------------------------------------------------')
predict_result = model(input) # 각 layer의 output 출력
after_softmax = F.softmax(predict_result, dim = 1)
print('----------------------------------------------------------------')
print("softmax 이후의 값", after_softmax)
print("걸리는 시간 :", time.time() - start) # 걸리는 시간 출력
print('----------------------------------------------------------------')
print(model.state_dict()) # model의 가중치 및 bias 출력

torch.save(model.state_dict(), 'save_weight.pt') # 훈련이 끝난 뒤 가중치 및 bias 저장
torch.save(model, 'save_model.pt') # 훈련이 끝난 모델 전체를 저장

sys.stdout.close() # 출력 값들을 txt 파일에 저장