from langchain_chroma import Chroma
from langchain_community.document_loaders import GitLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field


from dotenv import load_dotenv
import os


load_dotenv()


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if os.path.exists("./chroma_db"):
    print("Loading from existing database")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
else:
    print("Creating new database")
    loader = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain",
        repo_path="./langchain",
        branch="langchain==0.2.13",
        file_filter=file_filter,
    )
    documents = loader.load()
    print(len(documents))
    db = Chroma.from_documents(documents, embeddings, persist_directory="./chroma_db")

retriever = db.as_retriever()
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template(
    """
  다음 문맥만을 고려해 질문에 답하세요.
  
  문맥: '''{context}'''
  
  질문: '''{question}'''
  """
)


class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(..., description="검색 쿼리 목록")


query_generation_prompt = ChatPromptTemplate.from_template(
    """
  질문에 대해 벡터 데이터베이스에서 관련 문서를 검색하기 위한 
  3개의 서로 다른 검색 쿼리를 생성하세요.
  거리 기반 유사성 검색의 한계를 극복하기 위해
  사용자의 질문에 대해 여러 관점을 제공하는 것이 목표입니다.
  
  질문: {question}
  """
)

query_generation_chain = (
    query_generation_prompt
    | model.with_structured_output(QueryGenerationOutput)
    | (lambda x: x.queries)
)

multi_query_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context": query_generation_chain | retriever.map(),
    }
    | prompt
    | model
    | StrOutputParser()
)

output = multi_query_rag_chain.invoke("LangChain에서 RAG에 대해서 상세하게 알려줘")
print(output)
