from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence, TypedDict
from langchain_core.tools import create_retriever_tool
from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import tools_condition


from dotenv import load_dotenv

load_dotenv()

# 크롤링할 웹 페이지 URL 목록
urls = [
    "https://finance.naver.com/",
    "https://finance.yahoo.com/",
    "https://finance.daum.net/",
]

# 각 URL에서 문서 로드
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# 문서 분할 설정
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# 벡터 스토어에 문서 추가
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()


# 에이전트 상태를 나타내는 데이터 구조 정의
class AgentState(TypedDict):
    # add_messages 함수는 업데이트가 어떻게 처리되어야 하는지 정의한다.
    # 기본값은 대체이다. add_messages는 "추가"라고 말한다.
    messages: Annotated[Sequence[BaseMessage], add_messages]


# 검색 도구 설정
retriever_tool = create_retriever_tool(
    retriever,
    "retriever_blog_posts",
    "네이버, 야후, 다음의 금융 관련 정보를 검색하고 반환한다.",
)
tools = [retriever_tool]

# Edges
def grade_documents(state) -> Literal["generate", "rewrite"]:
  """
  검색된 문서가 질문과 관련이 있는지 평가한다.
  
  Args:
    state (messages): 현재 상태 
  
  Returns:
    str: 문서의 관련성에 따라 다음 노드 결정 ("generate" 또는 "rewrite")
  
  """
  print("---문서 관련성 평가---")
  
  # 데이터 모델 정의
  class grade(BaseModel):
    """관련성 평가를 위한 이진 점수."""
    binary_score: str = Field(description="관련성 점수 'yes' 또는 'no'")
    
  # LLM 모델 정의 
  model = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
  
  # LLM에 데이터 모델 적용
  llm_with_tool = model.with_structured_output(grade)
  
  prompt = P