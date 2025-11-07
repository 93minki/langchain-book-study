from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

response = client.responses.create(
  model="gpt-4o-mini",
  input="아내가 먹고 싶어한 음식이 뭐야?",
  tools=[{
    "type": "file_search",
    "vector_store_ids": ["vs_690c5b1cf20c81919fb793b2464415e6"]
  }]
)

print(response)