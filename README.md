
# Streamlit과 Upstage 솔루션 기반 Chatbot 생성

이 프로젝트는 **Streamlit과 LangChain**을 활용하여 사용자와 대화하고, 이전 대화 내용을 기억하는 챗봇을 생성합니다.

---

## 🎯 연구 목표 🎯

1. **LLM 활용** : Upstage의 Solar LLM(`solar-1-mini-chat`)

2. **대화 기억 방식** : Streamlit의 `st.session_state` 사용(No RAG)

3. **감성 분석 기능**  
  3-1) 챗봇의 답변을 HuggingFace의 `pipeline`, `AutoTokenizer`, `AutoModelForSequenceClassification` 을 사용하여 감정 분석, 토큰화, 긍정/부정 감성 분석을 진행하고, 결과(라벨, 점수)를 채팅창에 표시.  
  3-2) 감성 분석 모델: `sangrimlee/bert-base-multilingual-cased-nsmc`  

4. **프롬프트 엔지니어링 강화**  
  4-1) 챗봇의 역할, 행동 지침, 부적절한 내용 제재, 프롬프트 인젝션 방어 원칙 등을 프롬프트 `System_Message`에 정의.  
  4-2) `temperature`, `max_tokens` 수치를 조정하여 자연스럽고 부드런운 언행이 표시될 수 있도록 조정.  

5. **대화 횟수 제한**  
  5-1) 불필요한 자원 낭비를 제한을 위해 사용자가 챗봇에게 보낼 수 있는 메시지 횟수를 제한하고, 남은 횟수를 사이드바에 Markdown으로 표시.
  
---

## ⚠️ 고지 및 한계 ⚠️

- 이 코드는 **비상업적, 개인 연구 목적**에 한해 활용되어야 하며, 대규모 운영, 자동화, 혹은 상업적 응용에 사용할 수 없습니다.

---

## 라이선스

본 프로젝트는 MIT 라이선스로 배포됩니다. 자세한 내용은 [LICENSE](./LICENSE)를 참조하십시오.
