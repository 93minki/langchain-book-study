from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

prompt_template = "이 음식 리뷰 '{review}'에 대해 '{rating1}'점 부터 '{rating2}'점까지의 평가를 해주세요"
prompt = PromptTemplate(
    input_variables=["review", "rating1", "rating2"], template=prompt_template
)

openai = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY, temperature=0.7)

chain = prompt | openai | StrOutputParser()


try:
    response = chain.invoke(
        {
            "review": "맛은 있엇지만 배달 포장이 부족하여 아쉬웠습니다.",
            "rating1": "1",
            "rating2": "5",
        }
    )
    print(f"평가 결과: {response}")
except Exception as e:
    print(f"Error: {e}")
