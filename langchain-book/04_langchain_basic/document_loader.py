from langchain_community.document_loaders import GitLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="langchain==0.2.13",
    file_filter=file_filter,
)

raw_docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

docs = text_splitter.split_documents(raw_docs)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

query = "AWS의 S3에서 데이터를 읽어 들이기 위한 Document Loader가 있나요?"

vector = embeddings.embed_query(query)

db = Chroma.from_documents(docs, embeddings)

retriever = db.as_retriever()

prompt = ChatPromptTemplate.from_template(
    """
  다음 문맥만을 바탕으로 질문에 답변해주세요.
  
  문맥: {context}
  
  질문: {question}
  """
)

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
