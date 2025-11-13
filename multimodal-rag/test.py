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


def extract_pdf_elements(path, fname):
    """
    PDF 파일에서 이미지, 테이블, 텍스트 블록을 추출한다.
    path: 파일 경로 (이미지 파일이 저장될 위치)
    fname: 파일 이름
    """

    return partition_pdf(
        filename=os.path.join(path, fname),
        languages=["kor", "eng"],
        extract_images_in_pdf=True,  # PDF 파일에서 이미지 추출
        infer_table_structure=True,  # PDF 파일에서 테이블 구조 추론
        chunking_strategy="by_title",  # 타이틀을 기준으로 텍스트 블록으로 분할
        max_characters=4000,  # 최대 4000자로 텍스트 블록 제한
        new_after_n_chars=3800,  # 3800자 이후에 새로운 블록 생성
        combine_text_under_n_chars=2000,  # 2000자 이하의 텍스트는 결합
        image_output_dir_path=path,  # 이미지가 저장될 경로 설정
    )


current_directory = os.getcwd()
fname = "invest.pdf"
fpath = os.path.join(current_directory, "multimodal-rag", "invest")
raw_pdf_elements = extract_pdf_elements(fpath, fname)


def dump(el, depth=0, max_chars=120):
    name = el.__class__.__name__
    cat = getattr(el, "category", None)
    txt = (getattr(el, "text", "") or "").replace("\n", " ")
    print("  " * depth + f"- {name} ({cat}) : {txt[:max_chars]}")
    children = getattr(el, "elements", None)
    if children:
        for ch in children:
            dump(ch, depth + 1, max_chars)


for i, el in enumerate(raw_pdf_elements):
    print(f"\n===== Element {i} =====")
    dump(el)
