# from langchain_core.retrievers import MultiQueryRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ensemble
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Loader
loader = PyPDFLoader("hybrid-search/sample.pdf")
pages = loader.load_and_split()
# print(pages)

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50,
)
texts = text_splitter.split_documents(pages)
# print(texts)

# Embedding
embedding_model = OpenAIEmbeddings()
# load it into Chroma
vectorstore = Chroma.from_documents(texts, embedding_model)
chroma_retriever = vectorstore.as_retriever(
    search_type="mmr", search_kwargs={"k": 1, "fetch_k": 4}
)

# Initialize the BM25 Retriever
bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 2  # Retrieve top 2 results

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.2, 0.8]
)

# query = "에코프로에 대해서 알려줘"

# docs = ensemble_retriever.invoke(query)

template = """
다음 맥락을 바탕으로 질문에 답변하세요:

{context}

질문: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


# Post-processing
def format_docs(docs):
    formatted = "\n\n".join(doc.page_content for doc in docs)
    return formatted


# Chain
rag_chain = (
    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
result = rag_chain.invoke("에코프로에 대해서 알려줘")
print(result)
