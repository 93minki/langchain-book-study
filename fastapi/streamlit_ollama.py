import streamlit as st
import requests

st.title("Langchain API Client Ollama")
st.write("주제에 맞는 소설과 시를 작성해주는 API 클라이언트 입니다.")

topic = st.text_input("주제를 입력하세요:")

option = st.radio("작성을 원하는 항목을 선택하세요:", ("소설", "시"))
# option = st.selectbox("작성을 원하는 항목을 선택하세요:", ("소설", "시"))

if st.button("작성 요청 보내기"):
    if option == "소설":
        response = requests.post(
            "http://localhost:8000/llama/novel/invoke", json={"input": {"topic": topic}}
        )
    else:
        response = requests.post(
            "http://localhost:8000/llama/poem/invoke", json={"input": {"topic": topic}}
        )

    if response.status_code == 200:
        st.write(f"### {option} 응답")
        content = response.json()
        st.write(content["output"])
        st.download_button(
            label="결과 다운로드",
            data=content["output"],
            file_name=f"{option}_result.txt",
            mime="text/plain",
        )
    else:
        st.write(f"{option} API 요청에 실패 했습니다.")
