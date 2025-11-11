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

# Make Splits
splits = text_splitter.split_documents(docs)

# Index
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# RAG Fusion 관련 검색 쿼리 생성
template = """
 당신은 주어진 하나의 질문을 기반으로 여러 검색 쿼리를 생성하는 유용한 조수입니다. \n 
 다음 질문과 관련된 여러 검색 쿼리를 생성하세요: {question} \n
 출력 (4개의 쿼리):
"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_rag_fusion
    | ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    | StrOutputParser()
    | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
)

# result = generate_queries.invoke(
#     {
#         "question": "코스피 지수의 향방에 대해서 알려줘 또한 외국인 투자자들의 매매 향방에 대해서도 알려줘"
#     }
# )
# print(result)


def reciprocal_rank_fusion(results: list[list], k=60, top_n=2):
    """
    여러 개의 순위가 매겨진 문서 리스트를 받아, RRF(Reciprocal Rank Fusion) 공식을 사용하여
    문서의 최종 순위를 계산하는 함수다. k는 RRF 공식에서 사용되는 선택적 파라미터이며,
    top_n은 반환할 우선순위가 높은 문서의 개수
    """
    fused_scores = {}

    for docs in results:
        # 리스트 내의 각 문서와 그 문서의 순위를 가져온다.
        for rank, doc in enumerate(docs):
            # 문서를 문자열 형식으로 직렬화하여 딕셔너리 키로 사용한다.(문서가 JSON형식으로 직렬화될 수 있다고 가정)
            doc_str = dumps(doc)
            # 해당 문서가 아직 딕셔너리에 없으면 초기 점수 0으로 추가한다.
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # 문서의 현재 점수를 가져온다. (이전에 계산된 점수)
            # RRF 공식을 사용하여 문서의 점수를 업데이트 한다. 1 / (순위 + k)
            fused_scores[doc_str] += 1 / (rank + k)
    # 문서들을 계산된 점수에 따라 내림차순으로 정렬하여 최종적으로 재정렬된 결과를 얻는다.
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return reranked_results[:top_n]


retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion

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

question = "코스피 지수의 향방에 대해서 알려줘 또한 외국인 투자자들의 매매 향방에 대해서도 알려줘"

result = final_rag_chain.invoke(question)
print(result)
