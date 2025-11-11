import bs4
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.load import dumps, loads

load_dotenv()

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
        "k": 3,
        "fetch_k": 4,
    },  # 상위 1개의 문서를 반환하지만, 고려할 문서는 4개로 설정
)

# Prompt Template
client = Client()
prompt = client.pull_prompt("sungwoo/ragbasic")

# LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


# Post-processing
def format_docs(docs):
    formatted = "\n\n".join(doc.page_content for doc in docs)
    return formatted


# Chain
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

template = """
당신은 AI 언어 모델 조수입니다. 당신의 임무는 주어진 사용자 질문에 대해 벡터 데이터베이스에서
관련 문서를 검색할 수 있도록 다섯 가지 다른 버전을 생성하는 것입니다.
사용자 질문에 대한 여러 관점을 생성함으로써, 거리 기반 유사성 검색의 한계를 극복하는 데 도움을 주는것이
목표입니다.
각 질문은 새 줄로 구분하여 제공하세요. 원본 질문: {question}
"""

prompt_perspective = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_perspective
    | ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    | StrOutputParser()
    | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
)

generated_query = generate_queries.invoke({"question": "집값의 향방?"})
print(generated_query)


def get_unique_union(documents: list[list]):
    """고유한 문서들의 합집합을 생성하는 함수"""

    # 리스트의 리스트를 평탄화하고, 각 문서를 문자열로 직렬화한다.
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]

    # 중복된 문서를 제거하고 고유한 문서만 남긴다.
    unique_docs = list(set(flattened_docs))

    # 고유한 문서를 원래의 문서 객체로 변환하여 반환
    return [loads(doc) for doc in unique_docs]


question = "한국의 경제 상황에 대해서 분석해봐 그리고 앞으로 경제 상황에대해서 예측해봐"
retrieval_chain = generate_queries | retriever.map() | get_unique_union

# docs = retrieval_chain.invoke({"question": question})
# print(len(docs))
# print(docs)

# RAG
template = """
다음 맥락을 바탕으로 질문에 답변하세요:

{context}

질문: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

final_rag_chain = (
    {"context": retrieval_chain, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = final_rag_chain.invoke(question)
print(result)
