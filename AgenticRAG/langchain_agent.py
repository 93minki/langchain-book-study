# Agents 생성을 위한 참조 Agent Excutor
from langchain_classic.agents import AgentExecutor

# 벡터 DB를 agent에 전달하기 위한 tool 생성
from langchain_classic.agents import create_openai_tools_agent

# langchainhub 에서 제공하는 prompt 사용
from langchain_classic import hub

# arXiv 논문 검색을 위한 tool 생성
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

# 벡터 DB 구축 및 검색 도구
from langchain_core.tools import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# langchain 공식 문서 검색을 위한 검색기 역할을 하는 벡터 DB 생성
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader

# agent tools 중 wikipedia 사용
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

# OpenAI LLM 설정
from langchain_openai import ChatOpenAI
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from dotenv import load_dotenv

load_dotenv()

openai = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# agent 시뮬레이션을 위한 prompt 참조
# hub에서 가져온 prompt를 agent에게 전달하기 위한 prompt 생성
prompt = hub.pull("hwchase17/openai-functions-agent")

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)

wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

print(wiki.name)

# 네이버 기사 내용을 가져와서 벡터 DB 생성
loader = WebBaseLoader("https://news.naver.com/")
docs = loader.load()
# 문서를 1000자의 덩어로 나누되, 각 덩어리의 200자 정도는 중첩되도록 설정
documents = RecursiveCharacterTextSplitter(
    chunk_size=3000, chunk_overlap=200
).split_documents(docs)

# 문서를 임베딩하고 FAISS 벡터 DB로 저장
vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vectordb.as_retriever()  # 벡터 DB를 검색기로 변환

# 검색기 객체 출력 확인
print(retriever)

retriever_tool = create_retriever_tool(
    retriever,
    "news_retriever",
    "네이버 뉴스 정보가 저장된 벡터 DB, 당일 기사에 대해 궁금하면 이 툴을 사용하세요!",
)

# 툴 이름 출력 확인
print(retriever_tool.name)

# arXiv API 설정: top_k_results = 결과 수, doc_content_chars_max = 문서 길이 제한
arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=200,
    load_all_available_meta=False,
)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
# arXiv 툴 이름 출력 확인
print(arxiv.name)

# agent가 사용할 tool을 정의하여 tools에 저장
tools = [wiki, retriever_tool, arxiv]
# agent llm 모델을 openai로 정의하고 tools, prompt를 입력하여 agent를 완성
agent = create_openai_tools_agent(llm=openai, tools=tools, prompt=prompt)

# agent Excute 정의 부분 verbose=True로 설정하면 agent 실행 과정을 출력한다.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# agent_result = agent_executor.invoke({"input": "llm 관련 최신 논문을 알려줘"})
agent_result = agent_executor.invoke({"input": "오늘 부동산 관련 주요 소식을 알려줘"})

print(agent_result)
