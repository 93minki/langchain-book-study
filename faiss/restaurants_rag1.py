from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 현재 파이썬 스크립트 실행 위치 반환
current_dir = os.path.dirname(os.path.abspath(__file__))
# 현재 파이썬 스크립트 실행 위치에 있는 "restaurant-faiss" 폴더 경로
restaurant_faiss = os.path.join(current_dir, "restaurant-faiss")
# TextLoader 클래스를 사용하여 "restaurant.txt" 라는 파일에서 텍스트를 로드합니다.
loader = TextLoader(f"{current_dir}/restaurants.txt")
# 파일의 내용을 document 객체로 로드합니다.
documents = loader.load()

# 텍스트를 300자 단위로 나누고, 연속된 청크 사이에 50자의 겹침을 두어 텍스트를 분할하는 text splitter 객체를 생성
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
# 로드된 문서를 지정된 크기와 겹침에 따라 더 작은 청크로 분할
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(docs, embeddings)
db.save_local(restaurant_faiss)
print("레스토랑 임베딩 저장 완료", restaurant_faiss)