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


def rerank(inp: dict[str, Any], top_n: int = 3) -> list[Document]:
    question = inp["question"]
    documents = inp["documents"]

    cohere_reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=top_n)
    return cohere_reranker.compress_documents(documents=documents, query=question)


rerank_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "documents": retriever,
    }
    | RunnablePassthrough.assign(context=rerank)
    | prompt
    | model
    | StrOutputParser()
)

rerank_rag_chain.invoke("LangChain의 개요를 알려줘")
