$$Phase 4 - Step 11$$

# 실무 0순위: LangChain으로 '내 블로그 읽어주는 AI 챗봇(RAG)' 만들기

지금까지 우리는 데이터를 다루는 법부터 머신러닝, PyTorch 딥러닝, 비전(CNN), 그리고 자연어 처리(Transformer)까지 AI의 방대한 역사를 압축해서 달려왔습니다.

그리고 오늘, 드디어 현재 AI 산업의 정점에 있는 거대 언어 모델(LLM, Large Language Model)을 실무에서 어떻게 활용하는지 다뤄볼 차례입니다. 챗GPT는 똑똑하지만 내 개인 정보나 사내 기밀 문서는 알지 못합니다. 이 문제를 해결하는 마법의 기술이 바로 RAG(검색 증강 생성)입니다.

오늘은 최고의 LLM 프레임워크인 LangChain(랭체인)을 활용해, 내 블로그 글을 읽고 답변해 주는 나만의 맞춤형 AI 챗봇을 만들어 보겠습니다!

## 1. RAG(Retrieval-Augmented Generation)란 무엇인가?
챗GPT에게 "내 블로그의 최신 글 요약해 줘"라고 하면 어떻게 대답할까요? 당연히 "저는 당신의 블로그 내용을 알 수 없습니다"라고 답하거나, 그럴싸한 거짓말(Hallucination, 환각)을 지어낼 것입니다.

LLM은 '머리 좋은 학생'이지만, 머릿속에 든 지식은 과거에 학습한 데이터뿐입니다.
RAG(검색 증강 생성)는 이 학생에게 '오픈북 테스트(Open-book test)'를 보게 하는 기술입니다.

1. **Retrieval (검색)**: 사용자가 질문하면, 먼저 내 문서(블로그 글, 회사 PDF 등) 창고에서 질문과 관련된 내용을 검색해서 찾아옵니다.
2. **Augmented (증강)**: 찾아온 내용을 프롬프트(질문)에 살을 붙여서 추가합니다.
3. **Generation (생성)**: LLM이 덧붙여진 문서를 읽고 정확하게 답변을 생성합니다.

이 기술 덕분에 LLM을 처음부터 다시 학습(Fine-tuning)시키지 않아도, 최신 정보나 비공개 데이터를 AI에게 똑똑하게 답변하게 만들 수 있습니다.

## 2. 실습 준비: LangChain과 벡터 DB 설치
Jupyter Notebook 터미널을 열고 필요한 라이브러리들을 설치합니다. 이 실습에서는 가장 대중적인 OpenAI의 API(gpt-3.5-turbo 또는 gpt-4o)를 사용하겠습니다.
```
pip install langchain langchain-openai chromadb tiktoken
```
(※ 로컬 무료 LLM을 쓰고 싶다면 ollama를 설치하여 대체할 수도 있지만, 실무 표준인 OpenAI API 기준으로 설명합니다. OpenAI API Key가 필요합니다.)

## 3. 실전 코딩: 나만의 RAG 시스템 구축하기
RAG 시스템은 크게 `데이터 준비` ➔ `벡터화 및 저장` ➔ `검색 및 답변` 3단계로 이루어집니다.

#### 3.1 데이터 준비 및 쪼개기 (Document Load & Split)
먼저 AI에게 읽힐 텍스트 데이터를 불러오고, AI가 소화하기 좋게 문단을 쪼개줍니다.
```python
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. API 키 설정 (실제 환경에서는 환경변수나 .env 파일에 숨겨야 합니다!)
os.environ["OPENAI_API_KEY"] = "sk-여러분-자신의-API-키를-입력하세요"

# 2. 임의의 블로그 글(텍스트) 데이터 생성
blog_content = """
[나의 AI 개발 일지]
오늘 Mac OS 환경에서 AI 개발 세팅을 마쳤다. Homebrew로 Miniforge를 설치하고, 
Python 3.10 가상 환경을 만들었다. PyTorch를 설치할 때 Apple Silicon의 MPS 가속을 
활용할 수 있도록 세팅한 것이 가장 뿌듯하다. 내일은 Pandas로 데이터를 전처리해 볼 예정이다.
"""

with open("my_blog.txt", "w", encoding="utf-8") as f:
    f.write(blog_content)

# 3. 데이터 불러오기
loader = TextLoader("my_blog.txt", encoding="utf-8")
docs = loader.load()

# 4. 텍스트 쪼개기 (Chunking)
# 책을 페이지 단위로 찢는 것과 같습니다. 검색을 빠르고 정확하게 하기 위함입니다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
splits = text_splitter.split_documents(docs)

print(f"총 {len(splits)}개의 조각으로 쪼개졌습니다.")
```


#### 3.2 임베딩(Embedding)과 벡터 DB(Vector DB) 저장
쪼개진 글 조각들을 컴퓨터가 이해할 수 있는 '숫자 좌표(벡터)'로 변환하여 창고에 저장합니다.

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 임베딩 모델 준비 (텍스트를 의미 기반의 숫자 벡터로 변환하는 AI)
embeddings = OpenAIEmbeddings()

# 2. 벡터 데이터베이스(ChromaDB)에 쪼개진 글(splits) 저장
# 실무에서는 이 DB를 서버에 구축해두고 수백만 건의 사내 문서를 넣어둡니다.
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# 3. 검색기(Retriever) 생성
# "유사도가 가장 높은 상위 2개의 조각만 찾아와!" 라고 세팅합니다.
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

print("벡터 DB에 내 블로그 글 저장 완료! 📚")
```


#### 3.3 프롬프트 조립 및 AI 답변 생성 (The RAG Chain)
이제 사용자가 질문하면, 벡터 DB에서 블로그 글을 찾아오고, 그것을 LLM에게 넘겨 답변을 작성하게 하는 파이프라인(Chain)을 만듭니다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. LLM 모델 불러오기 (머리 좋은 학생 입장!)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 2. 프롬프트 템플릿(지시서) 작성
# {context}에는 검색해 온 블로그 글이, {question}에는 사용자의 질문이 들어갑니다.
template = """
당신은 친절한 블로그 어시스턴트입니다. 
아래 제공된 블로그 글(Context)을 바탕으로 사용자의 질문(Question)에 답변해주세요.
만약 주어진 글에 내용이 없다면, "블로그에 해당 내용이 없습니다."라고 솔직하게 답변하세요.

Context: {context}

Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(template)

# 3. 조각들을 하나의 텍스트로 합쳐주는 도우미 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. 랭체인(LangChain) 파이프라인(Chain) 구성 - (가장 우아한 부분!)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("RAG 챗봇 시스템 구축 완료! 🚀")
```

#### 3.4 챗봇과 대화해 보기!
```
# 질문 1: 내 블로그 내용에 있는 질문
question1 = "블로그 주인이 오늘 어떤 가상 환경을 만들었어?"
print(f"👤 사용자: {question1}")
print(f"🤖 AI 답변: {rag_chain.invoke(question1)}\n")

# 질문 2: 내 블로그에 없는 질문 (환각 억제 테스트)
question2 = "블로그 주인의 취미는 뭐야?"
print(f"👤 사용자: {question2}")
print(f"🤖 AI 답변: {rag_chain.invoke(question2)}")
```


$$예상 실행 결과$$
```
👤 사용자: 블로그 주인이 오늘 어떤 가상 환경을 만들었어?
🤖 AI 답변: 블로그 주인은 오늘 Python 3.10 버전으로 가상 환경을 만들었습니다.

👤 사용자: 블로그 주인의 취미는 뭐야?
🤖 AI 답변: 블로그에 해당 내용이 없습니다.
```
완벽합니다! AI가 엉뚱한 소리(환각)를 하지 않고, 오직 우리가 제공한 블로그 문서에 기반해서 똑똑하게 답변하는 것을 확인할 수 있습니다.

## 마치며 (Phase 4 완료)

축하합니다! 방금 여러분은 수많은 기업들이 사내 지식 검색 시스템, 고객 센터 챗봇 등을 만들 때 사용하는 RAG 기술의 핵심 아키텍처를 직접 구현하셨습니다.

LangChain의 놀라운 점은, 오늘 작성한 이 짧은 코드에서 `TextLoader`를 `PyPDFLoader`로 바꾸면 PDF 요약 봇이 되고, `WebBaseLoader`로 바꾸면 특정 웹사이트를 읽어주는 봇으로 순식간에 탈바꿈한다는 것입니다.

지금까지 우리는 훌륭한 AI 모델들을 '만드는' 방법을 배웠습니다. 하지만 내 노트북(Jupyter) 안에서만 도는 AI는 반쪽짜리에 불과합니다.

마지막 **Phase 5 (Step 12~13)** 에서는 우리가 만든 이 멋진 AI를 다른 사람들도 인터넷으로 접속해서 써볼 수 있도록 서비스로 띄우는 MLOps와 서버 배포(FastAPI, Docker)의 세계로 진입하겠습니다!
