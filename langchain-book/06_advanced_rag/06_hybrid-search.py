from langchain_chroma import Chroma
from langchain_community.document_loaders import GitLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from typing import Any
from langchain_cohere import CohereRerank
from langchain_community.retrievers import BM25Retriever, TavilySearchAPIRetriever
from enum import Enum

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

chroma_retriever = retriever.with_config({"run_name": "chroma_retriever"})
bm25_retriever = BM25Retriever.from_documents(documents).with_config(
    {"run_name": "bm25_retriever"}
)

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template(
    """
  다음 문맥만을 고려해 질문에 답하세요.
  
  문맥: '''{context}'''
  
  질문: '''{question}'''
  """
)


class Route(str, Enum):
    langchain_document = "langchain_document"
    web = "web"


class RouteOutput(BaseModel):
    route: Route


route_prompt = ChatPromptTemplate.from_template(
    """
  질문에 답변하기 위해 적절한 Retriever를 선택하세요.
  
  질문: {question}
  """
)


route_chain = (
    route_prompt | model.with_structured_output(RouteOutput) | (lambda x: x.route)
)


def routed_retriever(inp: dict[str, Any]) -> list[Document]:
    question = inp["question"]
    route = inp["route"]

    if route == Route.langchain_document:
        return langchain_document_retriever.invoke(question)
    elif route == Route.web:
        return web_retriever.invoke(question)

    raise ValueError(f"Unknown route: {route}")


route_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "route": route_chain,
    }
    | RunnablePassthrough.assign(context=routed_retriever)
    | prompt
    | model
    | StrOutputParser()
)

route_rag_chain.invoke("오늘 서울의 날씨는?")
