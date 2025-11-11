import bs4
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.load import dumps, loads
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ensemble
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


### INDEXING ###

loader = WebBaseLoader(
    web_paths=("https://news.naver.com/section/101",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("sa_text", "sa_item_SECTION_HEADLINE"),
        )
    ),
)

docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50,
)

# Make Splits
splits = text_splitter.split_documents(docs)

# Index
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
chroma_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 1, "fetch_k": 4},
)

bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = 2

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.2, 0.8]
)

query = "향후 집값에 대해서 알려줘"

docs = ensemble_retriever.invoke(query)

template = """
당신은 AI 언어 모델 조수입니다. 당신의 임무는 주어진 사용자 질문에 대해 벡터 데이터베이스에서
관련 문서를 검색할 수 있도록 다섯 가지 다른 버전을 생성하는 것입니다. 사용자 질문에 대한 여러 
관점을 생성함으로써, 거리 기반 유사성 검색의 한계를 극복하는 데 도움을 주는 것이 목표입니다.
각 질문은 새 줄로 구분하여 제공하세요. 원본 질문: {question}
"""
prompt_perspective = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_perspective
    | ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
    # | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
)


def reciprocal_rank_fusion(results: list[list], k=60, top_n=2):
    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return reranked_results[:top_n]


retrieval_chain_rag_fusion = (
    generate_queries | ensemble_retriever.map() | reciprocal_rank_fusion
)

question = "향후 집값에 대해서 알려줘"
docs = retrieval_chain_rag_fusion.invoke({"question": question})

# RAG
template = """
다음 맥락을 바탕으로 질문에 답변하세요:

{context}

질문: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

final_rag_chain = (
    {"context": retrieval_chain_rag_fusion, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = final_rag_chain.invoke({"question": question})
print(result)
