from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [("system", "사용자가 입력한 요리의 레시피를 생각해 주세요."), ("human", "{dish}")]
)

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

output_parser = StrOutputParser()

chain = prompt | model | output_parser

# for chunk in chain.stream({"dish": "카레"}):
#     print(chunk, end="", flush=True)

outputs = chain.batch([{"dish": "카레"}, {"dish": "파스타"}])
print(outputs)
