import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client

from dotenv import load_dotenv

load_dotenv()

import os

os.environ["USER_AGENT"] = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
)

#### INDEXING ####

# Load Documents
loader = WebBaseLoader(
    web_paths=("https://news.naver.com/section/101",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=("sa_text", "sa_item_SECTION_HEADLINE"))
    ),
)
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, chunk_overlap=50
)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(
    search_type="mmr",  # MMR 알고리즘을 사용하여 검색
    search_kwargs={
        "k": 1,
        "fetch_k": 4,
    },  # 상위 1개의 문서를 반환하지만, 고려할 문서는 4개로 설정
)

client = Client()
prompt = client.pull_prompt("sungwoo/ragbasic")

# LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


# Post-processing
def format_docs(docs):
    formatted = "\n\n".join(doc.page_content for doc in docs)
    return formatted


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
result = rag_chain.invoke("국채 관련한 정보를 알려줘")
print(result)
