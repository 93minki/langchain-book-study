import requests

openai_endpoint = "http://localhost:8000/openai"
llama_endpoint = "http://localhost:8000/llama"

response1 = requests.post(
    f"{openai_endpoint}/novel/invoke", json={"input": {"topic": "행복에대해서"}}
)
response2 = requests.post(
    f"{openai_endpoint}/poem/invoke", json={"input": {"topic": "행복에대해서"}}
)

response3 = requests.post(
    f"{llama_endpoint}/novel/invoke", json={"input": {"topic": "행복에대해서"}}
)
response4 = requests.post(
    f"{llama_endpoint}/poem/invoke", json={"input": {"topic": "행복에대해서"}}
)

print(response1.json())
print("\n")
print(response2.json())
print("\n")
print(response3.json())
print("\n")
print(response4.json())
