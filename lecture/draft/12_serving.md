$$Phase 5 - Step 12$$

# AI 모델을 세상 밖으로: FastAPI로 딥러닝 모델 서빙(Serving)하기
지금까지 우리는 훌륭한 성능을 자랑하는 여러 AI 모델들을 직접 만들어 보았습니다. 그런데 한 가지 문제가 있습니다. 우리가 만든 이 똑똑한 AI는 현재 '**내 노트북의 Jupyter 환경**' 안에서만 동작한다는 것입니다.

만약 여러분이 스마트폰 앱 개발자나 웹 프론트엔드 개발자 친구에게 "내가 텍스트 감성 분석 AI를 만들었으니 네 앱에 넣어봐!"라고 한다면, 친구는 이렇게 물을 것입니다.
**"그래서 API 주소가 뭔데?"**

오늘은 Jupyter Notebook을 벗어나, 우리가 만든 AI 모델을 누구나 인터넷을 통해 접속해서 사용할 수 있도록 만들어주는 모델 서빙(Model Serving)의 기초를 배워보겠습니다.

## 1. 모델 서빙(Model Serving)과 REST API
AI 모델 서빙이란, 학습이 완료된 AI 모델을 서버에 올려두고 사용자(또는 다른 프로그램)의 요청(Request)을 받아 예측 결과(Response)를 반환하는 과정입니다. 이때 가장 표준적으로 사용되는 소통 방식이 REST API입니다.

- **사용자**: "이 문장 긍정이야 부정이야?" (데이터 전송)
- **API 서버**: "잠시만, AI에게 물어볼게." ➔ 모델 추론 ➔ "긍정(Positive)이고 확률은 99%야." (결과 반환)

파이썬에서 이런 API 서버를 만들 때 과거에는 Flask나 Django를 많이 썼지만, 최근 AI 업계의 압도적인 표준은 FastAPI입니다. 이름처럼 처리 속도가 엄청나게 빠르고 코드가 직관적이기 때문입니다.

## 2. 실습 준비: FastAPI 설치하기
새로운 파이썬 파일(예: `main.py`)을 만들고 코드를 작성해야 하므로, 이번엔 Jupyter Notebook이 아닌 VS Code 같은 코드 에디터를 사용하는 것을 추천합니다.

먼저 터미널(`ai_env` 가상환경)을 열고 FastAPI와 서버 구동을 위한 uvicorn을 설치합니다.
```
pip install fastapi uvicorn pydantic transformers
```
(※ 이번 실습에서는 Step 10에서 썼던 Hugging Face의 트랜스포머 파이프라인을 재사용합니다.)

## 3. 실전 코딩: 15줄로 끝내는 AI 서버 만들기
폴더에 `main.py`라는 파이썬 파일을 새로 만들고, 아래의 코드를 작성합니다.

```python
# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# 1. FastAPI 앱 초기화
app = FastAPI(title="나만의 AI 감성 분석 API")

# 2. AI 모델 로드 (서버가 켜질 때 한 번만 메모리에 올립니다)
print("AI 모델을 불러오는 중입니다...")
classifier = pipeline("sentiment-analysis")
print("모델 로드 완료!")

# 3. 데이터 형식 정의 (사용자가 서버로 보낼 데이터 구조)
# Pydantic을 사용하면 데이터 형식을 엄격하고 안전하게 관리할 수 있습니다.
class UserRequest(BaseModel):
    text: str

# 4. API 엔드포인트(URL 주소) 만들기
@app.post("/predict")
def predict_sentiment(request: UserRequest):
    # 사용자가 보낸 텍스트(request.text)를 AI 모델에 넣고 추론
    result = classifier(request.text)[0]
    
    # 결과를 JSON 형태로 깔끔하게 반환
    return {
        "input_text": request.text,
        "sentiment": result['label'],
        "confidence_score": round(result['score'], 4)
    }
```

이 짧은 코드가 우리가 만든 AI 모델을 전 세계와 연결해 줄 서버 프로그램의 전부입니다!

## 4. API 서버 실행 및 테스트하기

#### 서버 켜기
터미널에서 `main.py`가 있는 폴더로 이동한 뒤, 아래 명령어를 입력합니다.
```
uvicorn main:app --reload
```

- `main`: 우리가 만든 파이썬 파일 이름 (`main.py`)
- `app`: 코드 안에서 만든 FastAPI 객체 이름
- `--reload`: 코드를 수정하면 서버를 자동으로 재시작해주는 옵션

터미널에 `Application startup complete.`라는 문구가 뜨면 성공입니다. 여러분의 Mac이 훌륭한 AI 서버로 변신했습니다!

#### Swagger UI로 인공지능 테스트하기
FastAPI의 가장 강력한 장점 중 하나는 API 테스트 페이지(Swagger UI)를 자동으로 만들어준다는 것입니다.

1. 웹 브라우저를 열고 `http://127.0.0.1:8000/docs`로 접속합니다.
2. 초록색 박스로 된 `POST /predict`를 클릭합니다.
3. 우측 상단의 `Try it out` 버튼을 누릅니다.
4. `Request body`의 "text" 부분을 원하는 문장으로 바꿉니다. (예: `"This Fastapi tutorial is amazing!"`)
5. 파란색 `Execute` 버튼을 누릅니다.

화면 아래쪽 Server response 부분에 AI가 분석한 JSON 결과가 출력되는 것을 볼 수 있습니다.
```
{
  "input_text": "This Fastapi tutorial is amazing!",
  "sentiment": "POSITIVE",
  "confidence_score": 0.9998
}
```

## 마치며
축하합니다! 방금 여러분은 인공지능 모델을 탑재한 백엔드 API 서버를 직접 구축했습니다. 이제 이 서버의 주소(`http://127.0.0.1:8000/predict`)만 프론트엔드 개발자에게 넘겨주면, 그들은 이 API를 호출해서 예쁜 웹사이트나 모바일 앱을 만들어낼 수 있습니다.

**하지만 아직 하나의 숙제가 남았습니다**. 이 서버는 지금 '내 노트북(Mac)'에서만 돌아갑니다. 내 노트북을 끄면 서비스도 죽어버립니다. 이를 클라우드(AWS, GCP 등)에 올려 365일 24시간 돌아가게 만들려면 어떻게 해야 할까요?

이 로드맵의 최종장, Step 13에서는 이 API 서버를 완벽하게 포장해서 어디서든 똑같이 실행할 수 있게 만들어주는 마법의 기술 'Docker(도커)'에 대해 알아보겠습니다!
