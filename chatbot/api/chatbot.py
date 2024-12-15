import requests
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Flask 서버 생성
app = Flask(__name__)

# GPT 모델과 Chroma DB 설정
llm = ChatOpenAI(
    openai_api_key="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",
    model_name='gpt-4o-mini'
)
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="안녕? 오늘 기분은 어때?"),
    AIMessage(content="I'm great thank you. How can I help you?"),
]

embed_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key="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")

persist_directory = "path/to/chroma_db"
db = Chroma.from_texts([], embed_model, persist_directory=persist_directory)
loaded_db = Chroma(persist_directory=persist_directory, embedding_function=embed_model)

# GitHub에서 텍스트 불러오기
github_raw_url = "https://raw.githubusercontent.com/noahon/24_2/refs/heads/main/translated_texts.txt"
response = requests.get(github_raw_url)

if response.status_code == 200:
    texts = response.text.splitlines()  # 파일 내용을 줄 단위로 분리
    texts = [text.strip() for text in texts if text.strip()]  # 빈 줄 제거
else:
    raise Exception(f"Failed to fetch file from GitHub. Status code: {response.status_code}")

# GitHub 데이터를 Chroma DB에 저장
db = Chroma.from_texts(texts, embed_model, persist_directory=persist_directory)
loaded_db = Chroma(persist_directory=persist_directory, embedding_function=embed_model)

# 카카오톡 웹훅 처리
@app.route("/kakao", methods=["POST"])
def kakao():
    # 카카오톡에서 전송된 메시지 받기
    data = request.json
    user_message = data.get("userRequest", {}).get("utterance", "")
    
    # 질문을 Chroma DB와 GPT 모델을 통해 처리하기
    def augment_prompt(query: str, previous_responses: str = ""):
        results = db.similarity_search(query, k=2)
        source_knowledge = "\n".join([x.page_content for x in results])
        augmented_prompt = f"""You are a helpful assistant. Answer the following query considering the previous conversation and context.

        Previous responses:
        {previous_responses}

        Contexts:
        {source_knowledge}

        Query: {query}"""
        return augmented_prompt
    
    # 지난 답변을 저장할 변수
    previous_responses = ""
    
    # GPT 모델에 쿼리 보내기
    if 'conversation' in data:
        previous_responses = data['conversation']  # 대화 내역이 있다면 그 내용 추가

    prompt = HumanMessage(content=augment_prompt(user_message, previous_responses))
    
    # GPT 모델에 쿼리 보내기
    messages = [
        HumanMessage(content=user_message),
        prompt
    ]
    
    response = llm.invoke(messages)

    
    # 응답 처리 (카카오톡 챗봇에 반환)
    return jsonify({
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": response
                    }
                }
            ]
        }
    })

# 서버 실행
if __name__ == "__main__":
    app.run(debug=True, port=5000)