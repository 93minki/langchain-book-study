import pytesseract
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
import json
from dotenv import load_dotenv
import base64

load_dotenv()


# GPT
gpt_model = ChatOpenAI(model="gpt-4o", temperature=0)

# Gemini
google_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Anthropic
anthropic_model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
)


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def solve_math_image(image_path, model):

    img_b64 = encode_image(image_path)

    prompt = """
너는 한국 중학교 수학 문제를 푸는 교사이다.
다음 이미지는 한 개의 수학 문제이다.

1. 이미지를 보고 문제를 정확히 텍스트로 옮겨 적어라.
2. 문제를 실제로 풀어 정답을 구해라.
3. 풀이 과정을 단계별로 간단히 정리해라.
4. 아래 JSON 형식으로만 출력해라.

{
  "problem_text": "...문제를 한국어 텍스트로 그대로 적기...",
  "curriculum": "예: 중1 1학기 함수",
  "type": "직선의 방정식, 그래프 해석 등",
  "concept": "기울기 부호, 오른쪽 위로 향하는 직선 등",
  "difficulty": "상/중/하 중 하나",
  "answer": "예: ㄴ, ㄷ, ㅁ",
  "solution_steps": "단계별 풀이 과정",
  "solution": "풀이 과정을 한국어로 단계별 요약"
}
"""

    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            },
        ]
    )

    result = model.invoke([msg])
    return result.content


print(solve_math_image("./math-image-vector/100367-web-obj-exam.png", gpt_model))


# def ocr_extract_text(img_path):
#     img = Image.open(img_path)
#     text = pytesseract.image_to_string(img, lang="kor")
#     return text


# def build_analysis_chain():
#     model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

#     parser = JsonOutputParser()

#     prompt = ChatPromptTemplate.from_template(
#         """
#       당신은 수학 문제 분석 전문가입니다.

#       아래 OCR로 추출한 수학 문제 내용을 기반으로 다음 정보를 JSON으로 정리하세요.

#       OCR 텍스트:
#       ---------
#       {ocr_text}
#       ---------

#       반드시 다음 키를 JSON 형식으로 출력하세요:
#       - curriculum: 어떤 학년/학기/단원인지
#       - type: 문제 유형 (예: 함수, 도형, 방정식, 비례식, 표해석 등)
#       - concept: 관련 개념
#       - difficulty: 난이도 (상/중/하)
#       - contains_table: 표가 있는지 (true/false)
#       - contains_formula: 수식 포함 여부(true/false)
#       - requires_calcuation: 계산 문제인지 (true/false)
#       - summary: 문제 요약
#       - answer: 최종 정답
#       - solution: 풀이 설명 요약

#       출력은 반드시 JSON만 출력하세요
#       {format_instructions}
#       """
#     )

#     prompt = prompt.partial(format_instructions=parser.get_format_instructions())
#     chain = prompt | model | parser
#     return chain


# if __name__ == "__main__":
#     img_path = "./math-image-vector/print-sbj-exam.png"

#     # OCR
#     ocr_text = ocr_extract_text(img_path)
#     print("\n======== OCR 결과 ========")
#     print(ocr_text)

#     # LangChain 분석
#     chain = build_analysis_chain()
#     result = chain.invoke({"ocr_text": ocr_text})

#     print("\n======== 분석 결과 ========")
#     print(json.dumps(result, ensure_ascii=False, indent=2))
