from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = init_chat_model(model="gpt-4o-mini", model_provider="openai")
result = model.invoke("hello")