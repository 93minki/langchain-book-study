from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
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


prompt = ChatPromptTemplate.from_template(
    """
  다음 문맥만을 고려해 질문에 답하세요.
  
  문맥: '''{context}'''
  
  질문: '''{question}'''
  
  """
)

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
retriever = db.as_retriever()

# chain = (
#     {"question": RunnablePassthrough(), "context": retriever}
#     | prompt
#     | model
#     | StrOutputParser()
# )

hypothetical_prompt = ChatPromptTemplate.from_template(
    """
  다음 질문에 한 문장으로 답하세요.
  질문: {question}
  """
)

hypothetical_chain = (
    hypothetical_prompt | model | StrOutputParser() | RunnableLambda(lambda x: (print(str(x)), x)[1])
)

hyde_rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context": hypothetical_chain | retriever,
    }
    | prompt
    | model
    | StrOutputParser()
)

output = hyde_rag_chain.invoke("LangChain의 개요를 알려줘")
print(output)
