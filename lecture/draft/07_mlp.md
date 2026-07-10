$$Phase 3 - Step 7$$

# 딥러닝의 꽃: PyTorch로 인공신경망(MLP) 만들고 손글씨 분류하기
이전 Step에서 우리는 PyTorch의 핵심인 '텐서(Tensor)'와 '자동 미분(Autograd)'을 배웠습니다. 오늘은 이 도구들을 조립하여 진짜 '인공신경망(Artificial Neural Network)'을 만들어 볼 차례입니다.

머신러닝(Step 3~5)에서는 우리가 데이터를 정형화(엑셀 표처럼)하고 공식을 적용했습니다. 하지만 딥러닝은 다릅니다. 인간의 뇌신경 구조를 모방한 다층 퍼셉트론(MLP, Multi-Layer Perceptron)을 이용해, 이미지 픽셀 덩어리를 통째로 집어넣고 기계가 스스로 패턴을 학습하게 만들 것입니다.

딥러닝의 영원한 첫 번째 과제, 'MNIST 손글씨 숫자 분류기'를 지금부터 밑바닥부터 구축해 보겠습니다!

## 1. 다층 퍼셉트론(MLP)이란?
우리 뇌는 수많은 '뉴런'들이 복잡하게 연결되어 정보를 처리합니다. 이를 컴퓨터로 모방한 것이 인경신경망입니다.
- **입력층 (Input Layer)**: 데이터를 받아들이는 곳입니다. (예: 이미지의 픽셀들)
- **은닉층 (Hidden Layer)**: 입력된 데이터에서 특징을 추출하고 복잡한 패턴을 학습하는 곳입니다. 이 은닉층이 깊어질수록 우리는 이를 '딥러닝(Deep Learning)'이라고 부릅니다.
- **출력층 (Output Layer)**: 최종 결과를 내놓는 곳입니다. (예: 0~9 중 어떤 숫자인지 확률 출력)

여기에 활성화 함수(Activation Function, 주로 ReLU 사용)를 더해주면, 단순한 직선(선형)을 넘어 구불구불하고 복잡한 곡선(비선형) 패턴까지 AI가 학습할 수 있게 됩니다.

## 2. 실전 코딩: 손글씨 분류기 만들기
Jupyter Notebook을 열고, PyTorch를 이용해 데이터를 불러오고 모델을 훈련하는 전체 파이프라인을 작성해 봅시다.

#### 2.1 데이터 준비 (Dataset과 DataLoader)
PyTorch는 이미지, 텍스트 등 방대한 데이터를 쉽게 쪼개서(Batch) 모델에 먹여줄 수 있는 훌륭한 도구를 제공합니다.
```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# 1. MNIST 데이터셋 다운로드 (Train/Test 분리)
# ToTensor()는 이미지를 0~1 사이의 숫자로 이루어진 텐서로 변환해 줍니다.
training_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

# 2. DataLoader 세팅
# 데이터를 한 번에 64개씩(배치 사이즈) 묶어서 모델에 공급합니다.
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

print(f"훈련 데이터 개수: {len(training_data)}")
print(f"테스트 데이터 개수: {len(test_data)}")
```

#### 2.2 신경망 모델 설계 (nn.Module)
가장 중요한 모델 구조 설계입니다. PyTorch에서는 항상 nn.Module을 상속받아 나만의 모델 클래스를 만듭니다.

```python
# 3. Mac GPU(MPS) 디바이스 설정
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"사용 중인 디바이스: {device}")

# 4. 신경망(MLP) 클래스 정의
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 28x28 사이즈의 2차원 이미지를 784개의 1차원 배열로 쭉 폅니다.
        self.flatten = nn.Flatten() 
        
        # 은닉층과 출력층을 쌓습니다. (Sequential로 묶으면 편합니다)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # 입력층 (784) -> 은닉층 1 (512)
            nn.ReLU(),             # 활성화 함수 (비선형성 부여)
            nn.Linear(512, 512),   # 은닉층 1 (512) -> 은닉층 2 (512)
            nn.ReLU(),             # 활성화 함수
            nn.Linear(512, 10)     # 은닉층 2 (512) -> 출력층 (10: 0~9 숫자)
        )

    # 데이터가 모델을 통과하는 흐름(순전파)을 정의합니다.
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 모델을 생성하고 GPU(MPS)로 보냅니다.
model = NeuralNetwork().to(device)
print(model)
```


#### 2.3 손실 함수와 옵티마이저 설정
모델이 얼마나 틀렸는지 채점하는 '손실 함수', 그리고 오차를 줄이기 위해 가중치를 업데이트하는 '옵티마이저'를 설정합니다.

```python
# 5. 채점 기준(Loss)과 최적화 방법(Optimizer) 
loss_fn = nn.CrossEntropyLoss() # 다중 분류에 가장 많이 쓰이는 손실 함수
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # 실무에서 가장 애용되는 Adam!
```


#### 2.4 모델 훈련 (Training Loop)
이제 이전에 배웠던 "**예측 -> 오차 계산 -> 자동 미분(역전파) -> 업데이트"** 과정을 코드로 엮어냅니다. 이 뼈대는 챗GPT를 만들 때도 똑같이 쓰입니다!

```python
# 6. 훈련 함수 정의
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # 모델을 훈련 모드로 설정
    
    for batch, (X, y) in enumerate(dataloader):
        # 데이터를 MPS(GPU)로 이동
        X, y = X.to(device), y.to(device)

        # 예측(Forward) 및 손실(Loss) 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파(Backpropagation) 및 가중치 업데이트
        optimizer.zero_grad() # 이전 기울기 초기화 (매우 중요!)
        loss.backward()       # 자동 미분 (Autograd 작동)
        optimizer.step()      # 가중치 업데이트

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# 7. 테스트(평가) 함수 정의
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # 모델을 평가 모드로 설정 (dropout 등 비활성화)
    test_loss, correct = 0, 0
    
    with torch.no_grad(): # 평가할 때는 미분(학습)이 필요 없으므로 메모리 절약!
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    print(f"테스트 결과: \n 정확도: {(100*correct):>0.1f}%, 평균 Loss: {test_loss:>8f} \n")

# 8. 본격적인 학습 시작 (3번 반복)
epochs = 3
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("딥러닝 학습 완료! 🎉")
```


#### 2.5 내 모델 직접 테스트해보기
AI가 얼마나 똑똑해졌는지, 실제 테스트 데이터의 이미지를 하나 뽑아서 예측 결과를 눈으로 확인해 봅시다.

```python
# 모델 평가 모드
model.eval()

# 테스트 데이터 하나 꺼내기 (0번째 이미지)
x, y = test_data[0][0], test_data[0][1]

with torch.no_grad():
    x = x.to(device) # 이미지를 GPU로
    pred = model(x)
    predicted_class = pred[0].argmax(0).item()
    
print(f"실제 숫자: {y}")
print(f"AI의 예측: {predicted_class}")

# 이미지 눈으로 확인하기
plt.imshow(x.cpu().squeeze(), cmap="gray")
plt.title(f"AI Prediction: {predicted_class}")
plt.axis("off")
plt.show()
```


## 마치며

축하합니다! 방금 여러분은 수만 개의 가중치(파라미터)를 가진 딥러닝 인공신경망을 직접 설계하고, Mac의 GPU를 활용해 학습까지 완수했습니다.

코드 라인이 조금 길어졌지만, `데이터 로드 -> 모델 설계 -> 훈련 루프`로 이어지는 이 파이프라인 구조는 어떤 최신 딥러닝 프로젝트를 하더라도 동일하게 적용되는 절대적인 템플릿입니다. 꼭 숙지해 두세요!

다음 Step 8에서는 오늘 무심코 지나갔던 딥러닝 튜닝의 핵심, 옵티마이저(Optimizer)와 손실 함수(Loss Function)에 대해 조금 더 깊게 파헤쳐 보겠습니다.
