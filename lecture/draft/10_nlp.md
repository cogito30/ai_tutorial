$$Phase 4 - Step 10$$

# 현대 AI의 언어 능력: 자연어 처리(NLP)와 트랜스포머(Transformer)의 혁명

지난 Step 9에서 우리는 CNN을 통해 AI에게 세상을 보는 '눈'을 달아주었습니다. 이제 AI에게 '언어(말과 글)'를 가르칠 차례입니다.

인간의 언어를 컴퓨터가 이해하고 처리하도록 만드는 분야를 자연어 처리(NLP, Natural Language Processing)라고 합니다. 번역기, 스팸 메일 필터링, 그리고 우리가 매일 쓰는 챗GPT까지 모두 NLP의 산물입니다.

오늘은 과거의 AI가 문장을 어떻게 읽었는지 알아보고, 현재 전 세계 AI의 판도를 뒤바꿔놓은 혁명적인 아키텍처 '트랜스포머(Transformer)'에 대해 알아보겠습니다.

## 1. 과거의 방식: 순환 신경망 (RNN과 LSTM)

언어 데이터(텍스트)는 이미지와 완전히 다른 특성이 있습니다. 바로 '순서(Sequence)'가 존재한다는 것입니다.
```
"나는 배가 고파서 ___을 먹었다."
```

빈칸에 들어갈 말을 맞추려면 앞의 단어들("배가 고파서")을 기억하고 있어야 합니다. 그래서 등장한 것이 RNN(Recurrent Neural Network)입니다. RNN은 단어를 앞에서부터 순서대로 하나씩 읽으며 기억(Hidden State)을 다음 단계로 넘겨줍니다.

🚨 **RNN의 치명적인 한계**

1. **건망증 (장기 의존성 문제)**: 문장이 길어지면 앞에 읽었던 단어를 까먹습니다.
2. **느린 속도**: 단어를 무조건 순서대로 하나씩 읽어야 하므로, GPU를 이용한 병렬 처리가 불가능합니다. 학습 속도가 처참하게 느립니다.

이 건망증을 조금 개선한 것이 LSTM(Long Short-Term Memory)이지만, 근본적인 '느린 속도' 문제는 해결하지 못했습니다.

## 2. 혁명의 시작: 트랜스포머 (Transformer)

2017년, 구글(Google) 연구진은 "Attention Is All You Need(어텐션만 있으면 돼)"라는 도발적인 제목의 논문을 발표합니다. 이 논문에서 제안한 아키텍처가 바로 트랜스포머(Transformer)입니다. (챗GPT의 'T'가 바로 이 Transformer입니다!)

#### 💡 트랜스포머의 핵심 마법: '셀프 어텐션(Self-Attention)'

트랜스포머는 RNN처럼 단어를 순서대로 읽지 않습니다. 문**장 전체의 단어를 한 번에 통째로 읽어 들입니다**. 그리고 각 단어가 문장 내의 **다른 모든 단어들과 얼마나 연관성이 있는지(Attention)를 수학적으로 계산**합니다.
```
"The bank of the river." (강의 둑)
"I deposited money in the bank." (나는 은행에 돈을 입금했다.)
```
똑같은 "bank"라는 단어라도, 트랜스포머는 주변에 "river"가 있는지 "money"가 있는지 한눈에 파악하여 문맥에 맞는 정확한 의미를 찰떡같이 알아냅니다.
- **압도적인 장점**: 단어를 한 번에 처리하므로 **GPU 병렬 처리**가 가능해졌고, 엄청나게 거대한 모델을 초고속으로 학습시킬 수 있게 되었습니다. 이것이 바로 **거대 언어 모델(LLM)** 탄생의 배경입니다.

## 3. 실전 코딩: Hugging Face로 트랜스포머 활용하기
컴퓨터 비전에 PyTorch의 torchvision이 있다면, 자연어 처리에는 Hugging Face(허깅페이스)가 있습니다. 허깅페이스는 전 세계의 AI 연구자들이 자신이 만든 트랜스포머 모델을 올려두고 공유하는 'AI계의 깃허브(GitHub)'입니다.

우리가 직접 트랜스포머를 밑바닥부터 짤 필요 없이, 허깅페이스 라이브러리를 통해 최고 성능의 모델을 단 세 줄이면 불러올 수 있습니다.

#### 3.1 준비물 설치
Jupyter Notebook 터미널에서 허깅페이스 핵심 라이브러리를 설치합니다.
```
pip install transformers
```

#### 3.2 감성 분석 (Sentiment Analysis) 모델 사용해보기
어떤 텍스트가 긍정적인지 부정적인지 판별하는 AI를 만들어보겠습니다. `pipeline` 함수를 사용하면 놀랍도록 쉽습니다.
```python
from transformers import pipeline

# 1. 감성 분석 파이프라인 불러오기
# 모델을 지정하지 않으면 기본적으로 영어 감성 분석에 특화된 트랜스포머 모델을 다운로드합니다.
classifier = pipeline("sentiment-analysis")

# 2. AI에게 문장 읽혀보기
sentences = [
    "I absolutely love this new Mac! The performance is outstanding.",
    "This tutorial is so hard to understand and confusing.",
    "The weather today is just okay, nothing special."
]

# 3. 결과 확인
results = classifier(sentences)

for sentence, result in zip(sentences, results):
    print(f"문장: {sentence}")
    print(f"➔ 예측: {result['label']} (확률: {result['score']*100:.2f}%)\n")
```


$$실행 결과 예시$$
```
예측: POSITIVE (확률: 99.98%)
예측: NEGATIVE (확률: 99.85%)
```

#### 3.3 한국어 텍스트 요약 모델 사용하기
이번엔 허깅페이스 허브에 한국어 데이터로 누군가 미리 학습시켜 둔(Pre-trained) '문서 요약' 모델을 가져와 보겠습니다.

```python
# 한국어 요약에 특화된 모델('gogamza/kobart-summarization')을 지정해서 불러옵니다.
summarizer = pipeline("summarization", model="gogamza/kobart-summarization")

text = """
인공지능(AI) 기술이 급속도로 발전함에 따라 다양한 산업 분야에서 AI의 도입이 가속화되고 있습니다. 
특히 자연어 처리 분야에서는 트랜스포머 아키텍처의 등장으로 챗봇, 번역, 문서 요약 등에서 인간과 
유사한 수준의 성능을 보여주고 있습니다. 하지만 일각에서는 AI의 편향성 문제와 일자리 감소 등에 
대한 우려의 목소리도 커지고 있어, 기술 발전과 함께 윤리적 가이드라인 마련이 시급하다는 지적이 
나오고 있습니다.
"""

# 요약 실행
summary = summarizer(text, max_length=50, min_length=10, do_sample=False)

print("📝 원본 요약 결과:")
print(summary[0]['summary_text'])
```

이처럼 실무에서의 NLP는 훌륭한 트랜스포머 모델을 허깅페이스에서 찾고, 우리의 문제에 맞게 파인튜닝(Fine-tuning)하거나 API로 연결하여 서비스에 녹여내는 방식으로 진행됩니다.

## 마치며

오늘 우리는 기존 RNN의 한계를 박살 내고 현대 AI의 기틀을 마련한 트랜스포머(Transformer)의 개념을 배웠습니다. '어텐션(Attention)' 메커니즘 덕분에 AI는 이제 문맥의 뉘앙스까지 이해하게 되었습니다.

이제 여러분은 비전(CNN)과 언어(Transformer)라는 AI의 두 가지 큰 축을 모두 이해하게 되었습니다!

대망의 다음 Step 11에서는 이 트랜스포머 기술의 결정체인 거대 언어 모델(LLM)을 직접 다루어 볼 것입니다. LangChain 프레임워크를 활용해 "내 블로그 문서를 읽고 답변해 주는 나만의 맞춤형 AI 챗봇(RAG 시스템)"을 구축해 보겠습니다!
