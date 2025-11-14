from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from langserve import add_routes
import uvicorn

from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Langchain Server", version="0.1.0", description="simple langchain API Server"
)

openAiModel = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
llamaModel = OllamaLLM(model="llama3.1:8b")

# 한국어 답변 보안을 위한 프롬프트 템플릿을 생성한다.
prompt = ChatPromptTemplate.from_template("한국어로 답변을 작성해줘{input}")

# 소설 작성용 프롬프트 템플릿을 생성한다.
prompt2 = ChatPromptTemplate.from_template(
    "주제에 맞는 소설을 작성해줘 500자 이내로 작성해줘 {topic}"
)

# 시 작성용 프롬프트 템플릿을 생성한다.
prompt3 = ChatPromptTemplate.from_template(
    "주제에 맞는 시를 작성해줘 200자 이내로 작성해줘 {topic}"
)

add_routes(
    app,
    prompt
    | openAiModel,  # ChatOpenAI 클래스의 인스턴스를 사용하여 OpenAI 모델과 상호작용한다.
    path="/openai",
)
add_routes(
    app,
    prompt
    | llamaModel,  # 이 경로로 요청이 들어오면 OllamaLLM 인스턴스를 통해 처리된다.
    path="/llama",
)
add_routes(
    app,
    prompt2 | openAiModel,  # 프롬프트 템플릿과 모델을 결합하여 처리한다.
    path="/openai/novel",  # 이 경로로 요청이 들어오면 소설을 작성한다.
)
add_routes(
    app,
    prompt3 | openAiModel,
    path="/openai/poem",
)
add_routes(
    app,
    prompt2 | llamaModel,
    path="/llama/novel",
)
add_routes(
    app,
    prompt3 | llamaModel,
    path="/llama/poem",
)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
