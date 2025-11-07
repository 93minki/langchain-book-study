from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))


# main()이라는 비동기 함수를 정의한다.
async def main():
    # 환경 변수에서 가져온 OpenAI API 키를 사용하여 OpenAIEmbeddings 클래스를 초기화
    embeddings = OpenAIEmbeddings()
    # 지정된 임베딩을 사용하여 로컬에 저장된 FAISS 인덱스를 로드한다.
    # allow_dangerous_deserialization=True 옵션은 역직렬화를 허용한다.
    load_db = FAISS.load_local(
        f"{current_dir}/restaurant-faiss",
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # 검색할 쿼리 문자열을 정의한다.
    query = "음식점의 룸 서비스는 어떻게 운영되나요?"
    # 'query' 변수는 사용자가 검색하려는 질문이나 문장을 담고 있습니다.
    # 'k=2'는 가장 유사한 문서 2개를 반환하도록 지정한다.
    result = load_db.similarity_search(query, k=2)
    # 검색 결과를 출력한다.
    print(result, "\n")

    # 쿼리를 임베딩 벡터로 변환한다.
    embedding_vector_query = embeddings.embed_query(query)
    print("Query Vector: ", embedding_vector_query)

    docs = await load_db.asimilarity_search_by_vector(embedding_vector_query)
    print(docs[0])


if __name__ == "__main__":
    asyncio.run(main())
