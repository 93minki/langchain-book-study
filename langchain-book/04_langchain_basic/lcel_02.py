from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain
from dotenv import load_dotenv
from operator import itemgetter


load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant."), ("human", "{input}")]
)

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

output_parser = StrOutputParser()


@chain
def upper(text: str) -> str:
    return text.upper()


chain = prompt | model | output_parser | upper

ai_message = chain.invoke("Hello, how are you?")
print(ai_message)
