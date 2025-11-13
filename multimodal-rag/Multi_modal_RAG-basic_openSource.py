from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv
import base64
import os
import requests
import json
import uuid
import io
import re
import time
from IPython.display import HTML, display
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from PIL import Image
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_experimental.open_clip import OpenCLIPEmbeddings

load_dotenv()


def extract_pdf_elements(path, fname):
    """
    PDF 파일에서 이미지, 테이블, 텍스트 블록을 추출한다.
    path: 파일 경로 (이미지 파일이 저장될 위치)
    fname: 파일 이름
    """

    return partition_pdf(
        filename=os.path.join(path, fname),
        extract_images_in_pdf=True,  # PDF 파일에서 이미지 추출
        infer_table_structure=True,  # PDF 파일에서 테이블 구조 추론
        chunking_strategy="by_title",  # 타이틀을 기준으로 텍스트 블록으로 분할
        max_characters=4000,  # 최대 4000자로 텍스트 블록 제한
        new_after_n_chars=3800,  # 3800자 이후에 새로운 블록 생성
        combine_text_under_n_chars=2000,  # 2000자 이하의 텍스트는 결합
        image_output_dir_path=path,  # 이미지가 저장될 경로 설정
    )


def categorize_elements(raw_pdf_elements):
    """
    PDF에서 추출한 요소들을 테이블과 텍스트로 분류합니다.
    raw_pdf_elements: unstructured.documents.elements 리스트
    """
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))  # 테이블 요소를 저장
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))  # 텍스트 요소를 저장
    return texts, tables


current_directory = os.getcwd()
# fname = "20250930_032910050110.pdf"
fname = "invest.pdf"
fpath = os.path.join(current_directory, "multimodal-rag", "invest")

# print("현재 스크립트의 위치:", current_directory)
# print("pdf 위치:", fpath)

raw_pdf_elements = extract_pdf_elements(fpath, fname)
texts, tables = categorize_elements(raw_pdf_elements)

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=2000, chunk_overlap=200
)

joined_texts = " ".join(texts)
texts_2k_token = text_splitter.split_text(joined_texts)

# print(len(texts_2k_token))
# print(len(texts))

# print(texts[0])
# print(type(texts))
# print(texts_2k_token[0])
# print(type(texts_2k_token))


def generate_text_summaries(texts, tables, summarize_texts=False):
    """
    텍스트 및 표 데이터를 요약하여 검색에 활용할 수 있는 요약본을 생성합니다.
    texts: 텍스트 리스트
    tables: 표 리스트
    summarize_texts: 텍스트 요약을 활성화할지 여부를 결정하는 불리언 값
    """

    # Prompt 영어 버전
    # prompt_text = """
    #   You are an assistant taksed with summarizing tables and text for retrieval.
    #   These summaries will be embedded and used to retrieve the raw text or table elements.
    #   Give a concise summary of the table or text that is well optimized for retrieval.
    #   Table of text: {element}
    # """

    prompt_text_kor = """
    당신은 표와 텍스트를 요약하여 검색에 활용할 수 있도록 돕는 도우미 입니다.
    이 요약본들은 임베딩되어 원본 텍스트나 표 요소를 검색하는 데 사용될 것입니다.
    주어진 표나 텍스트의 내용을 검색에 최적화된 간결한 요약으로 작성해주세요. 
    요약할 표 또는 텍스트: {element}
    """

    prompt = ChatPromptTemplate.from_template(prompt_text_kor)
    # 모델: GPT-4o-mini or Llama3.1 모델 사용
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    # llamaModel = OllamaLLM(model="llama3.1:8b")
    # summarize_chain = {"element": lambda x: x} | prompt | llamaModel | StrOutputParser()

    text_summaries = []
    table_summaries = []

    # 텍스트 요약을 활성화한 경우
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 1})
        # 텍스트 요약을 사용하지 않는 경우
    elif texts:
        text_summaries = texts

    # 테이블 데이터 요약
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 1})

    return text_summaries, table_summaries


text_summaries, table_summaries = generate_text_summaries(
    texts_2k_token, tables, summarize_texts=True
)


def encode_image(image_path):
    """이미지를 base64 문자열로 변환"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt, max_retries=5):
    """이미지 요약 생성 (Rate limit 에러 처리 포함)"""
    # chat = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)
    chat = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)

    for attempt in range(max_retries):
        try:
            msg = chat.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                },
                            },
                        ]
                    )
                ]
            )
            print(msg)
            return msg.content
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait_time = (2**attempt) * 0.5  # 지수 백오프: 0.5초, 1초, 2초, 4초, 8초
                print(
                    f"Rate limit 에러 발생. {wait_time:.1f}초 후 재시도... (시도 {attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)
            else:
                print(f"에러 발생: {e}")
                raise
    raise Exception(f"최대 재시도 횟수({max_retries})를 초과했습니다.")


def image_summarize_llava(img_base64, prompt):
    """LLaVA를 사용하여 이미지 요약 생성"""
    payload = {"model": "llava:7b", "prompt": prompt, "images": [img_base64]}
    response = requests.post("http://localhost:11434/api/generate", json=payload)

    print("Status Code: ", response.status_code)

    if response.status_code == 200:
        try:
            # 응답을 줄 단위로 처리
            full_response = ""
            for line in response.iter_lines():
                if line:
                    json_line = json.loads(line)
                    if "response" in json_line:
                        full_response += json_line["response"]
            print(full_response)
            return full_response
        except json.JSONDecodeError as e:
            return f"JSON 파싱 오류: {str(e)}\n 응답 내용: {response.text}"
    else:
        return f"Error: {response.status_code}, {response.text}"


def generate_image_summaries(path):
    """
    이미지의 요약과 base64 인코딩 문자열을 생성
    path: Unstructured에 의해 추출된 .jpg 파일의 경로
    """

    # Base64로 인코딩된 이미지를 저장할 리스트
    img_base64_list = []

    # 이미지 요약을 저장할 리스트
    image_summaries = []

    # Prompt_kor 한국어
    prompt_kor = """
  You are an assistant tasked with summarizing images for retrieval.
  These summaries will be embedded and used to retrieve the raw image.
  Provide a concise summary of the image that is well optimized for retrieval.
  The summary should be written in Korean (Hangul).
  """

    # Prompt 영어
    prompt = """
  You are an assistant tasked with summarizing images for retrieval.
  These summaries will be embedded and used to retrieve the raw image.
  Provide a concise summary of the image that is well optimized for retrieval.
  """

    # 주어진 경로에서 파일 목록을 가져와 정렬한 후, 각 파일을 처리한다.
    for img_file in sorted(os.listdir(path)):
        # 파일이 .jpg, .png, .jpeg 확장자 중 하나일 경우에만 처리한다.
        if img_file.endswith((".jpg", ".png", ".jpeg")):
            # 파일의 전체 경로를 생성한다
            img_path = os.path.join(path, img_file)
            # 이미지 파일을 Base64로 인코딩하여 문자열로 변환한다.
            base64_image = encode_image(img_path)
            # 인코딩된 Base64 문자열을 리스트에 추가한다.
            img_base64_list.append(base64_image)

            # LLaVA 모델을 사용하여 이미지 요약을 생성하고 리스트에 추가한다.
            # 또한, openai api의 image_summarize(base64_image, prompt_kor)를 사용하여 대체할 수도 있다.
            # 여기서는 한국어로 요약된 결과를 사용하고 있다.
            image_summaries.append(image_summarize(base64_image, prompt_kor))
            # Rate limit을 피하기 위해 요청 간 대기 시간 증가 (1초)
            time.sleep(1.0)
    return img_base64_list, image_summaries


# 현재 작업 디렉터리 경로를 찾는다.
current_directory = os.getcwd()

# 현재 디렉터리를 기준으로 'figures' 폴더 경로를 설정한다.
figures_directory = os.path.join(current_directory, "figures")

print(figures_directory)

# 이미지 요약 생성
img_base64_list, image_summaries = generate_image_summaries(figures_directory)
# len(img_base64_list)
# len(image_summaries)

# print(img_base64_list[0])
# print(image_summaries[0])


def create_multi_vector_retriever(
    vectorstore,
    text_summaries,
    texts,
    table_summaries,
    tables,
    image_summaries,
    images,
):
    """
    요약된 내용을 인덱싱하지만, 실제 검색 시 원본 이미지를 반환하는 검색기 생성
    """

    # 저장소 초기화
    store = InMemoryStore()
    id_key = "doc_id"

    # 다중 벡터 검색기 생성
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # 벡터 저장소와 문서 저장소에 문서를 추가하는 헬퍼 함수
    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]

        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    # 텍스트, 테이블, 이미지 추가
    # text_summaries 가 비어 있지 않으면 추가
    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    # table_summaries 가 비어 있지 않으면 추가
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    # image_summaries 가 비어 있지 않으면 추가
    if image_summaries:
        add_documents(retriever, image_summaries, images)

    return retriever


# OpenCLIPEmbeddings 모델을 사용한 임베딩 설정
embedding = OpenCLIPEmbeddings()

# 요약본을 인덱싱하는 데 사용할 벡터 저장소 설정
vectorstore = Chroma(collection_name="mm_rag_finace", embedding_function=embedding)

retriever_multi_vector_img = create_multi_vector_retriever(
    vectorstore,
    text_summaries,
    texts_2k_token,
    table_summaries,
    tables,
    image_summaries,
    img_base64_list,
)


def plt_img_base64(img_base64):
    """base64로 인코딩된 문자열을 이미지로 표시"""
    # Base64 문자열을 소스로 하는 HTML img 태그 생성
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}"/>'
    # HTML을 렌더링하여 이미지를 표시
    display(HTML(image_html))


def looks_like_base64(sb):
    """문자열이 base64 형식인지 확인"""
    if not isinstance(sb, str):
        return False
    # base64 문자열은 최소 길이가 있어야 함.
    if len(sb) < 100:
        return False
    # 공백과 줄바꿈 제거 후 확인
    sb_clean = sb.replace(" ", "").replace("\n", "")

    return re.match(r"^[A-Za-z0-9+/]+=*$", sb_clean) is not None


def is_image_data(b64data):
    """
    base64 데이터가 이미지인지 확인 (데이터 시작 부분을 검사)
    """
    if not isinstance(b64data, str):
        return False

    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        # 공백과 줄바꿈 제거
        b64data_clean = b64data.replace(" ", "").replace("\n", "")
        # base64 디코딩 시도 (validate=True로 유효성 검사)
        decoded = base64.b64decode(b64data_clean, validate=True)
        if len(decoded) < 8:
            return False
        header = decoded[:8]
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception as e:
        # 디버깅을 위해 예외 정보 출력 (선택사항)
        # print(f"is_image_data 예외: {type(e).__name__}: {e}")
        return False


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Base64로 인코딩된 이미지를 크기 조정
    """
    # Base64 문자열을 디코딩
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # 이미지 크기 조정
    resized_img = img.resize(size, Image.LANCZOS)

    # 크기 조정된 이미지를 bytes 버퍼에 저장
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # 크기 조정된 이미지를 Base64로 인코딩
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def split_image_text_types(docs):
    """
    문서에서 이미지와 텍스트를 분리
    """
    b64_images = []
    texts = []
    for i, doc in enumerate(docs):
        # 문서 유형이 Document일 경우 page_content 추출
        if isinstance(doc, Document):
            doc_content = doc.page_content
        else:
            doc_content = doc

        # 문자열이 아니면 텍스트로 처리
        if not isinstance(doc_content, str):
            texts.append(doc_content)
            continue

        # base64 형식인지 확인
        is_base64 = looks_like_base64(doc_content)
        if is_base64:
            # 이미지 데이터인지 확인
            is_img = is_image_data(doc_content)
            if is_img:
                try:
                    # 공백과 줄바꿈 제거 후 리사이즈
                    doc_content_clean = doc_content.replace(" ", "").replace("\n", "")
                    resized_image = resize_base64_image(
                        doc_content_clean, size=(1300, 600)
                    )
                    b64_images.append(resized_image)
                    # 디버깅 로그 (선택사항)
                    # print(f"[이미지 감지] 문서 {i}: 이미지로 분류됨 (길이: {len(doc_content)})")
                    continue
                except Exception as e:
                    # 이미지 처리 실패 시 텍스트로 처리 (원본 데이터 보존)
                    print(f"[이미지 처리 실패] 문서 {i}: {type(e).__name__}: {e}")
                    texts.append(doc_content)
            else:
                # base64 형식이지만 이미지가 아닌 경우 텍스트로 처리
                texts.append(doc_content)
        else:
            # base64 형식이 아닌 경우 텍스트로 처리
            texts.append(doc_content)

    # 디버깅 로그 (선택사항)
    # print(f"[분류 결과] 이미지: {len(b64_images)}개, 텍스트: {len(texts)}개")
    return {"images": b64_images, "texts": texts}


def img_prompt_func(data_dict):
    """
    주어진 맥락을 하나의 문자열로 결합하여 처리
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # 이미지가 있는 경우 메시지에 추가
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    # 분석할 텍스트 추가
    text_message = {
        "type": "text",
        "text": (
            "You are financial analyst tasking with providing investment advice.\n"
            "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
            "Use this information to provide investment advice related to the user question. \n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}\n\n"
            "Please provide the final answer in Korean(hangul)."
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


def multi_modal_rag_chain(retriever):
    """
    다중 모드 RAG 체인 생성
    """

    # 다중 모드 LLM 설정
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=1024)
    llava_model = OllamaLLM(model="llava:7b")
    # RAG 파이프라인 설정
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        # | llava_model
        | model
        | StrOutputParser()
    )
    return chain


def korean_convert_rag():
    """
    영어 텍스트를 한국어로 변환하는 RAG 체인
    """
    # or gpt-4 oepn-ai 모델 선택부분은 공식 홈페이지에서 확인 가능합니다. 현재는 gpt-3.5-turbo 모델을 사용하겠습니다.
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    # llamaModel = OllamaLLM(model="llama3.1:8b")

    # 프롬프트 템플릿 설정
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that translates English to Korean.",
            ),
            ("human", "Translate the following English text to Korean: {english_text}"),
        ]
    )

    # RAG 파이프라인 설정
    chain = (
        {"english_text": RunnablePassthrough()}
        | prompt
        # | llamaModel  # 또는 다른 모델 사용 가능
        | model
        | StrOutputParser()
    )

    return chain


chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)

# openSource model 사용시 답변이 영어로 나온다면 korean_convert_rag 체인을 생성하여 RAG 진행
korean_convert_rag = korean_convert_rag()
final_multimodal_rag = chain_multimodal_rag | korean_convert_rag

# query = "주가변동률과 가장 관련 있는 자료를 찾아줘"
# docs = retriever_multi_vector_img.invoke(query)

# 반환된 결과 개수 확인
# print(len(docs))
# 첫 번째 문서를 이미지로 표시한다.
# plt_img_base64(docs[0])

# 저장된 이미지 확인
# print(plt_img_base64(img_base64_list[3]))

# 해당 이미지의 요약 확인
# print(image_summaries[3])

query = "코스피와 관련된 전망을 종합적으로 알려줘"
# chain_multimodal_rag.invoke(query)

print(final_multimodal_rag.invoke(query))
