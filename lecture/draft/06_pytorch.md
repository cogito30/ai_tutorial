$$Phase 3 - Step 6$$

# 딥러닝의 시작: PyTorch 텐서(Tensor)와 Autograd 완벽 이해
지금까지 우리는 머신러닝의 뼈대(회귀, 분류)와 실전 무기(랜덤 포레스트, XGBoost)를 다루었습니다. 머신러닝 모델들은 정형(표) 데이터에서 엄청난 위력을 발휘합니다.

하지만 이미지, 음성, 자연어(텍스트) 같은 복잡한 '비정형 데이터'를 다루기 위해서는 인간의 뇌 구조를 모방한 **인공신경망(Artificial Neural Networks)**, 즉 딥러닝(Deep Learning)이 필요합니다.

오늘부터 우리는 딥러닝 연구 및 실무 점유율 압도적 1위 프레임워크인 PyTorch(파이토치)를 사용합니다. 딥러닝의 세계로 들어가는 첫 관문, 텐서(Tensor)와 Autograd(자동 미분)에 대해 알아보겠습니다.

## 1. 텐서(Tensor)란 무엇인가?
텐서(Tensor)라는 말이 아주 거창하게 들리지만, 사실 우리가 Step 2에서 배웠던 **NumPy의 다차원 배열(Array)과 거의 똑같습니다**. 

- 0차원 텐서: 스칼라 (그냥 숫자 하나, 예: 3)
- 1차원 텐서: 벡터 (숫자의 나열, 예: \[1, 2, 3\])
- 2차원 텐서: 행렬 (표 형태, 흑백 이미지 데이터 등)
- 3차원 이상의 텐서: 고차원 텐서 (컬러 이미지, 비디오 데이터 등)

**그럼 굳이 왜 NumPy를 안 쓰고 텐서를 쓸까요?**
- 가장 큰 이유는 **"GPU 가속"** 때문입니다. NumPy는 CPU에서만 연산되지만, PyTorch의 텐서는 수천 개의 코어를 가진 GPU에 올라가 엄청난 속도의 병렬 처리가 가능합니다.

## 2. 텐서 생성과 Mac GPU(MPS) 연동 실습
Jupyter Notebook을 열고 직접 코드를 쳐보면서 텐서와 친해져 봅시다!
```python
import torch
import numpy as np

# 1. 텐서 만들기
# 리스트에서 만들기
t1 = torch.tensor([[1, 2], [3, 4]])
print(f"기본 텐서:\n{t1}")

# 난수로 채워진 2x3 텐서 만들기
t2 = torch.rand(2, 3)
print(f"\n랜덤 텐서:\n{t2}")
print(f"텐서의 크기(Shape): {t2.shape}")

# 2. NumPy와 완벽한 호환성
np_array = np.array([5, 6, 7])
# NumPy -> PyTorch Tensor
tensor_from_np = torch.from_numpy(np_array) 
# PyTorch Tensor -> NumPy
back_to_np = tensor_from_np.numpy() 

# 3. 🚀 Mac Apple Silicon GPU(MPS) 활용하기 (가장 중요!)
# 딥러닝 학습 시 데이터를 GPU 메모리로 보내야 연산이 빨라집니다.
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("\n✅ MPS(Mac GPU) 사용 가능!")
else:
    device = torch.device("cpu")
    print("\n⚠️ CPU 사용")

# 텐서를 생성하면서 바로 MPS로 보내거나
x = torch.ones(3, 3, device=device)
# 만들어진 텐서를 MPS로 보냅니다 (.to 함수 사용)
y = torch.tensor([1.0, 2.0]).to(device)

print(f"x가 있는 위치: {x.device}")
print(f"y가 있는 위치: {y.device}")
```

실무에서는 무거운 이미지 데이터나 텍스트 데이터를 통째로 `tensor.to('cuda' 또는 'mps')` 명령어를 통해 GPU로 올려서 학습을 진행합니다.

## 3. 딥러닝의 마법: 자동 미분 (Autograd)
Step 3에서 '경사하강법(Gradient Descent)'을 밑바닥부터 구현할 때, 오차를 줄이기 위해 기울기(Gradient, 미분값)를 구하는 공식을 우리가 직접 코딩했던 것 기억하시나요?

딥러닝의 신경망은 수백만, 수십억 개의 파라미터(w, b)가 미로처럼 얽혀 있습니다. 인간이 이것을 일일이 손으로 미분하는 것은 불가능합니다.

PyTorch의 진정한 강력함은 Autograd 엔진에서 나옵니다. 텐서에 `requires_grad=True`라는 옵션만 켜주면, 네트워크가 아무리 복잡해도 파이토치가 알아서 연산 과정을 추적하고 미분값을 자동으로 계산해 줍니다.

수식 $y = 3x^2 + 4x$ 를 예로 들어 자동 미분을 확인해 보겠습니다.
(이 식을 $x$에 대해 미분하면 $y' = 6x + 4$ 가 됩니다. 만약 $x=2$ 라면, 기울기는 $6(2) + 4 = 16$ 입니다.)

```python
# 1. 텐서 생성 및 기울기 추적 활성화 (requires_grad=True)
# x = 2.0 인 텐서를 만듭니다.
x = torch.tensor(2.0, requires_grad=True)

# 2. 어떤 수식(함수) 통과시키기 (Forward Pass)
y = 3 * (x ** 2) + 4 * x

print(f"x의 값: {x.item()}")
print(f"y의 연산 결과: {y.item()}")

# 3. 마법의 주문! 역전파를 통한 자동 미분 계산 (Backward Pass)
y.backward()

# 4. x의 기울기(Gradient) 확인
print(f"수학적으로 계산한 기울기: 16.0")
print(f"PyTorch가 찾아낸 기울기(x.grad): {x.grad.item()}")
```

출력 결과가 16.0으로 완벽하게 일치하는 것을 볼 수 있습니다!

우리가 앞으로 만들 거대한 인공신경망에서도 PyTorch는 모델의 예측 결과(Forward)를 내놓고, `loss.backward()` 단 한 줄만 호출하면 모델 내의 수만 개의 파라미터(w, b)들의 미분값을 순식간에 .grad에 저장해 줍니다. 개발자는 그저 이 기울기를 따라 파라미터를 업데이트(경사하강법)만 해주면 됩니다.

## 마치며
오늘 배운 **"텐서(Tensor)를 만들고, GPU(MPS)에 올리고, Autograd로 미분한다."** 이 세 가지는 PyTorch의 심장이자 알파와 오메가입니다.

기초를 튼튼하게 다졌으니, 다음 Step 7에서는 드디어 이 텐서들을 엮어서 실제 뇌의 뉴런과 같은 인공신경망(MLP - Multi Layer Perceptron)을 만들고, 손글씨 숫자를 분류하는 첫 번째 딥러닝 모델을 밑바닥부터 훈련시켜 보겠습니다!
