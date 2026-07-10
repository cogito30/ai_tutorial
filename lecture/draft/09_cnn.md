$$Phase 4 - Step 9$$

# 인공지능이 눈을 뜨다: CNN과 전이 학습(Transfer Learning)으로 개/고양이 분류하기
지금까지 우리는 엑셀 같은 표 데이터(머신러닝)와 손글씨 숫자 같은 흑백의 단순한 이미지(MLP)를 다루어 보았습니다. 하지만 여러분의 스마트폰 앨범에 있는 고해상도의 컬러 사진들을 MLP로 학습시키려면 어떻게 될까요?

이미지의 픽셀 수가 너무 많아 메모리가 터져버리거나, 고양이가 사진 구석으로 조금만 이동해도 AI가 전혀 알아보지 못하는 참사가 발생합니다.

이러한 한계를 극복하고 현대 컴퓨터 비전(Computer Vision)의 혁명을 일으킨 기술이 바로 CNN(합성곱 신경망)입니다. 오늘은 CNN의 핵심 원리를 알아보고, 실무의 치트키인 '전이 학습'을 통해 강력한 개와 고양이 분류기를 만들어 보겠습니다.

## 1. CNN (Convolutional Neural Networks)의 두 가지 마법
CNN은 이미지를 1차원으로 길게 펴서 학습하던 기존 방식(MLP)과 달리, 이미지의 2차원 공간 형태를 유지한 채로 특징을 추출합니다.

1. **합성곱(Convolution) - "특징(Feature) 찾아내기"**
- 돋보기(필터/커널)를 이미지 위로 슬라이딩하면서 훑습니다.
- 첫 번째 층에서는 선, 질감, 색상 같은 단순한 특징을 찾고, 신경망이 깊어질수록 귀, 눈, 꼬리 같은 복잡한 형태를 조립해서 인식합니다.

2. **풀링(Pooling) - "핵심만 남기고 요약하기"**
- 이미지의 크기를 반으로 줄이면서 가장 강한 특징(주로 Max Pooling)만 남깁니다.
- 덕분에 연산량이 확 줄어들고, 고양이가 사진 정중앙에 있든 구석에 있든 유연하게 인식(위치 불변성)할 수 있게 됩니다.

## 2. 실무의 치트키: 전이 학습 (Transfer Learning)
실무에서 이미지를 분류할 때, 처음부터 CNN 모델을 설계하고 학습시키는 경우는 거의 없습니다. 시간이 너무 오래 걸리고 방대한 데이터가 필요하기 때문입니다.

대신 우리는 **거인의 어깨 위에 올라탑니다**. 구글, 메타 같은 빅테크 기업들이 수백만 장의 이미지(ImageNet)로 미리 똑똑하게 학습시켜 놓은 거대한 모델들(ResNet, VGG 등)을 무료로 다운로드해서 사용합니다.

이 똑똑한 뇌를 가져와서, **우리가 원하는 과제(개와 고양이 분류)에 맞게 마지막 출력 부분만 살짝 고쳐서 재학습**시키는 것을 전이 학습(Transfer Learning)이라고 합니다.

## 3. 실전 코딩: 전이 학습으로 개/고양이 분류기 만들기
Jupyter Notebook을 열고, PyTorch가 제공하는 가장 유명한 모델 중 하나인 `ResNet18`을 활용해 봅시다.

(※ 실습을 위해서는 `train/dogs`, `train/cats` 폴더에 각각 개와 고양이 이미지가 정리된 데이터셋이 필요합니다. Kaggle의 'Dogs vs. Cats' 데이터셋 구조를 가정하고 코드를 작성했습니다.)

#### 3.1 데이터 전처리 및 로더 설정
컬러 이미지는 크기가 제각각이므로, 모델에 넣기 전에 크기를 맞추고 정규화하는 작업이 필수입니다.
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

# 1. Mac GPU(MPS) 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"사용 장치: {device}")

# 2. 이미지 전처리 (Transforms)
# 학습 데이터는 모델이 강건해지도록 이미지를 자르고 뒤집는 증강(Augmentation)을 추가합니다.
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), # 224x224 크기로 자르고 리사이즈
        transforms.RandomHorizontalFlip(), # 좌우 반전
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet 표준 정규화
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 3. 데이터셋 및 데이터로더 불러오기 (폴더 구조 기반)
# (주의: 실제 로컬에 데이터 폴더가 있어야 코드가 작동합니다.)
import os
data_dir = './data/dogs_vs_cats' # 데이터셋 경로

try:
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
                   for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"클래스 종류: {class_names}") # ['cats', 'dogs']
except FileNotFoundError:
    print("⚠️ 데이터 폴더를 찾을 수 없어 모델 세팅만 진행합니다.")
```


#### 3.2 사전 학습된 모델(ResNet) 불러오고 튜닝하기
수백만 장의 이미지를 본 똑똑한 눈(CNN 특징 추출기)은 그대로 얼려두고, 마지막 판단을 내리는 머리(분류기)만 떼어내어 개/고양이(2개 클래스) 전용으로 갈아 끼웁니다.

```python
# 4. 사전 학습된 ResNet18 모델 불러오기
# 최신 PyTorch에서는 weights 매개변수를 사용하여 명시적으로 가져옵니다.
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# 5. 기존 파라미터 얼리기 (Freeze)
# "너희는 이미 똑똑하니까 더 이상 학습하지(업데이트되지) 마!"
for param in model.parameters():
    param.requires_grad = False

# 6. 마지막 분류기(Fully Connected Layer) 층 교체
# 원래 ResNet18은 1000개의 사물을 맞추도록 설계되어 있습니다. (model.fc.out_features == 1000)
# 이것을 우리의 목적에 맞게 2개(개, 고양이)로 바꿉니다. 
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2) # 입력은 그대로, 출력은 2개로!

# 교체된 새로운 마지막 층은 학습이 되도록 requires_grad가 기본적으로 True로 설정됩니다.
model = model.to(device)

print("모델 세팅 완료!")
```

#### 3.3 훈련 준비 및 학습 (마지막 층만!)
기존 수십 겹의 층은 학습하지 않고 오직 방금 갈아 끼운 model.fc 층만 훈련시키면 되므로, 학습 속도가 엄청나게 빠릅니다.

```python
# 7. 손실 함수와 옵티마이저 (마지막 fc 층의 파라미터만 업데이트!)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

print("훈련 준비 완료! 데이터셋이 있다면 아래와 같은 방식으로 학습합니다.")
print("""
# --- 훈련 루프 예시 ---
# model.train()
# for inputs, labels in dataloaders['train']:
#     inputs, labels = inputs.to(device), labels.to(device)
#     optimizer.zero_grad()
#     outputs = model(inputs)
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()
""")
```

단 10~20줄의 코드만으로, 여러분은 세상의 온갖 사물을 구분할 줄 아는 인공지능을 가져와 '최고 성능의 개/고양이 판독기'로 개조하는 데 성공했습니다.

## 마치며

오늘 배운 전이 학습(Transfer Learning)은 컴퓨터 비전 실무 프로젝트의 90% 이상을 차지하는 절대적인 방법론입니다. 자동차 파손 인식, 의료용 X-ray 질병 진단, 불량품 검출 등 어떤 비전 프로젝트를 하더라도 오늘 짠 코드에서 `데이터셋`과 `마지막 클래스 개수(2)`만 바꿔주면 곧바로 최고 수준의 성능을 낼 수 있습니다.

이제 이미지(눈)를 다루었으니, 다음 Step 10에서는 인간의 언어(입과 귀)를 다루는 자연어 처리(NLP)와 트랜스포머(Transformer)의 세계로 떠나보겠습니다!
