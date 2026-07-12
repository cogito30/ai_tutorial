$$ \[번외 편 - 심화\] $$

# LLM의 진화: 스스로 생각하고 일하는 'AI 에이전트(Agent)' 구축하기

우리는 지난 Step 11에서 내 블로그 문서를 읽고 답변하는 똑똑한 RAG 챗봇을 만들었습니다.
하지만 RAG 챗봇에게 이런 질문을 던지면 어떻게 될까요?
```
👤 사용자: "오늘 서울 날씨를 검색해 보고, 섭씨온도를 화씨로 변환해서 알려줘."
```
RAG 챗봇은 문서 창고(벡터 DB)에 저장된 과거의 지식만 뒤질 뿐, **스스로 인터넷을 검색하거나 계산기를 두드리는 능력은 없습니다**. 결국 "문서에 내용이 없습니다"라며 포기하고 말 것입니다.

단순히 묻는 말에 대답만 하는 수동적인 챗봇을 넘어, **목표를 주면 스스로 계획을 세우고, 필요한 도구(인터넷, 계산기 등)를 사용해 임무를 완수하는 능동적인 인공지**능. 이것이 바로 현재 전 세계 AI 업계가 열광하는 'AI 에이전트(Agent)'입니다.

오늘은 LangChain을 활용해 스스로 생각하고 일하는 나만의 AI 에이전트를 만들어 보겠습니다!

## 1. AI 에이전트(Agent)의 3가지 핵심 요소
에이전트는 LLM을 단순한 '텍스트 생성기'가 아니라 '추론 엔진(두뇌)'으로 사용합니다.

1. **두뇌 (LLM)**: 사용자의 명령을 분석하고, "어떤 순서로 일을 처리할지" 계획을 세웁니다.
2. **손과 발 (Tools)**: 에이전트가 사용할 수 있는 도구들입니다. 인터넷 검색, 파이썬 코드 실행기, 계산기, 사내 DB 접근 권한 등이 될 수 있습니다.
3. **생각의 흐름 (ReAct 방식)**: 에이전트는 Reasoning(생각)과 Acting(행동)을 반복합니다.

- 생각: "서울 날씨를 알아야겠다. 검색 도구를 쓰자."
- 행동: \[웹 검색: 서울 날씨\] ➔ 결과(Observation): 섭씨 20도.
- 생각: "섭씨 20도를 화씨로 바꿔야 하네. 계산기 도구를 쓰자."
- 행동: \[계산기: 20 * 1.8 + 32\] ➔ 결과: 68.
- 최종 답변: "오늘 서울은 화씨 68도입니다."

## 2. 실습 준비: 필수 라이브러리 설치
Jupyter Notebook 터미널을 열고, 에이전트 구동에 필요한 라이브러리들을 설치합니다. 이번 실습에서는 무료로 쉽게 붙일 수 있는 '위키백과 검색'과 '수학 계산기' 도구를 에이전트에게 쥐여줄 것입니다.
```
pip install langchain langchain-openai wikipedia numexpr langchainhub
```
(※ wikipedia는 위키백과 검색을 위한 패키지, numexpr는 LLM의 수학 계산을 돕는 패키지입니다.)

## 3. 실전 코딩: 수학과 검색이 가능한 에이전트 만들기
새로운 Jupyter 셀을 열고 아래 코드를 작성해 봅시다. 최신 LangChain의 AgentExecutor를 활용한 표준 방식입니다.
```python
import os
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools, AgentExecutor, create_openai_tools_agent
from langchain import hub

# 1. API 키 세팅 (실제로는 .env 파일 등을 사용해 안전하게 관리하세요)
os.environ["OPENAI_API_KEY"] = "sk-여러분-자신의-API-키를-입력하세요"

# 2. 에이전트의 두뇌(LLM) 준비
# 에이전트는 정확한 추론이 생명이므로 temperature를 0으로 설정하여 창의성보다는 논리성을 극대화합니다.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 3. 에이전트에게 쥐여줄 도구(Tools) 장착
# "wikipedia"(위키백과 검색)와 "llm-math"(LLM 전용 계산기)를 도구 상자에 넣습니다.
tools = load_tools(["wikipedia", "llm-math"], llm=llm)

# 4. 에이전트용 프롬프트 가져오기
# 허깅페이스처럼 랭체인 허브(LangChain Hub)에는 검증된 프롬프트들이 있습니다.
prompt = hub.pull("hwchase17/openai-tools-agent")

# 5. 에이전트와 실행기(Executor) 생성
# LLM(두뇌), Tools(손발), Prompt(지시서)를 하나로 묶어줍니다.
agent = create_openai_tools_agent(llm, tools, prompt)

# verbose=True로 설정하면 에이전트가 생각하고 행동하는 과정을 터미널에서 생생하게 볼 수 있습니다!
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("🤖 스스로 일하는 AI 에이전트 준비 완료!")
```

## 4. 에이전트 작동 테스트 (생각의 과정 엿보기)
이제 아주 까다로운 질문을 던져보겠습니다. 단순한 검색 한 번으로는 풀 수 없고, "검색 후 ➔ 연산"을 거쳐야만 하는 문제입니다.
```python
# LLM이 가장 취약한 '최신 정보 검색'과 '수학 연산'이 결합된 질문
question = "손흥민 선수의 출생 연도를 위키백과에서 찾고, 현재 연도(2026년)를 기준으로 그의 나이를 계산해서 알려줘."

print(f"👤 사용자: {question}\n")

# 에이전트에게 임무 하달!
response = agent_executor.invoke({"input": question})

print(f"\n🤖 최종 답변: {response['output']}")
```

#### 🔍 에이전트의 놀라운 실행 로그 (Verbose Output)
코드를 실행하면, 에이전트가 단번에 대답하는 것이 아니라 아래처럼 혼잣말하며 문제를 해결해 나가는 소름 돋는(?) 과정을 볼 수 있습니다.
```
> Entering new AgentExecutor chain...

Invoking: `wikipedia` with `{'query': 'Son Heung-min'}`
[Action] 위키백과 검색 도구 사용 ➔ "손흥민" 검색

Observation: Son Heung-min (Korean: 손흥민; born 8 July 1992)... (검색 결과)
[Thought] "아, 검색 결과를 보니 손흥민은 1992년생이구나. 이제 2026년에서 1992년을 빼서 나이를 구해야겠다."

Invoking: `Calculator` with `2026 - 1992`
[Action] 계산기 도구 사용 ➔ 2026 - 1992 입력

Observation: Answer: 34
[Thought] "계산 결과가 34로 나왔다. 이제 최종 답변을 작성해야겠다."

손흥민 선수는 1992년에 태어났으며, 현재 연도인 2026년을 기준으로 계산하면 그의 나이는 34세입니다.

> Finished chain.
```
단순한 앵무새 같았던 AI가 스스로 "무엇을 모르는지 파악 ➔ 도구를 사용해 정보 획득 ➔ 도구를 사용해 가공 ➔ 최종 결론 도출"이라는 인간의 업무 프로세스를 그대로 모방해 낸 것입니다!

## 5. 무궁무진한 에이전트의 확장성

우리는 이번 실습에서 단 2개의 도구(위키백과, 계산기)만 주었습니다. 하지만 실무에서는 개발자가 직접 파이썬 함수를 짜서 '커스텀 도구(Custom Tools)'로 만들어 에이전트에게 쥐여줄 수 있습니다.

- send_email(to, title, body) 함수를 만들어 주면 ➔ **"이메일 자동 발송 비서"**
- execute_sql(query) 함수를 만들어 주면 ➔ **"스스로 DB를 조회하고 분석하는 데이터 분석가"**
- control_smart_home(device, action) 함수를 주면 ➔ **"자비스(J.A.R.V.I.S) 같은 스마트홈 집사"**

에이전트 기술의 발전은 이제 막 시작되었습니다. 이 원리를 이해하셨다면, 단순히 AI를 '사용하는' 사람을 넘어 AI에게 '직업을 부여하는' 진정한 AI 엔지니어로 거듭나실 수 있습니다.

## 마치며

이로써 기초 파이썬부터 딥러닝, LLM RAG, 그리고 대망의 AI 에이전트 구축까지의 기나긴 여정을 모두 마쳤습니다! 여러분의 노트북 안에서 스스로 생각하고 도구를 사용하는 에이전트가 돌아가는 걸 보니 어떤가요? 미래가 성큼 다가온 것이 느껴지실 겁니다.

이제 여러분만의 창의적인 아이디어를 코드로 구현해 볼 시간입니다. 그동안 배운 지식을 바탕으로 멋진 AI 서비스를 세상에 내놓으시길 진심으로 응원합니다! 🚀
