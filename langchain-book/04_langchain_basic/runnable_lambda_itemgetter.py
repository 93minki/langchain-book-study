from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from operator import itemgetter

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
output_parser = StrOutputParser()

optimistic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 낙관주의자입니다. 사용자의 입력에 대해 낙관적인 의견을 제공하세요",
        ),
        ("human", "{topic}"),
    ]
)
optimistic_chain = optimistic_prompt | model | output_parser

pessimistic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 비관주의자입니다. 사용자의 입력에 대해 비관적인 의견을 제공하세요",
        ),
        ("human", "{topic}"),
    ]
)
pessimistic_chain = pessimistic_prompt | model | output_parser

synthesize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 객관적 AI 입니다. {topic}에 대한 두 가지 의견을 종합하세요.",
        ),
        (
            "human",
            "낙관적 의견: {optimistic_opinion} \n 비관적 의견: {pessimistic_opinion}",
        ),
    ]
)

synthesize_chain = (
    RunnableParallel(
        {
            "optimistic_opinion": optimistic_chain,
            "pessimistic_opinion": pessimistic_chain,
            "topic": itemgetter("topic"),
        }
    )
    | synthesize_prompt
    | model
    | output_parser
)

output = synthesize_chain.invoke({"topic": "생성형 AI의 진화에 관해"})
print(output)
