from langchain_community.document_loaders import GitLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import nest_asyncio
import pickle
from ragas.testset import TestsetGenerator
from ragas.testset.transforms.default import default_transforms
from ragas.testset.transforms.extractors import HeadlinesExtractor
from ragas.testset.transforms.splitters import HeadlineSplitter


from dotenv import load_dotenv
import os


load_dotenv()
nest_asyncio.apply()


def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")


if os.path.exists("./documents.pkl"):
    print("Loading saved documents")
    with open("./documents.pkl", "rb") as f:
        documents = pickle.load(f)
else:
    print("Loading documents from Git")
    loader = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain",
        repo_path="./langchain",
        branch="langchain==0.2.13",
        file_filter=file_filter,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    with open("./documents.pkl", "wb") as f:
        pickle.dump(documents, f)
    print("Documents saved to documents.pkl")


for document in documents:
    document.metadata["filename"] = document.metadata["source"]


generator = TestsetGenerator.from_langchain(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    embedding_model=OpenAIEmbeddings(model="text-embedding-3-small"),
)

base_transforms = default_transforms(
    documents=list(documents),  # 이미 있는 LangChain Document 리스트
    llm=generator.llm,  # from_langchain이 감싼 ragas LLM
    embedding_model=generator.embedding_model,
)

# 2) HeadlinesExtractor / HeadlineSplitter 제거
custom_transforms = [
    t
    for t in base_transforms
    if not isinstance(t, (HeadlinesExtractor, HeadlineSplitter))
]

testset = generator.generate_with_langchain_docs(
    documents,
    testset_size=4,
    transforms=custom_transforms,
)

testset.to_pandas().to_csv("ragas_testset.csv", index=False)
print("Testset saved to ragas_testset.csv")
