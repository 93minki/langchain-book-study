from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
import json
import logging
import uvicorn

from crew import create_crew

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def serialize_object(obj):
    if isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, dict):
        return obj
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        return str(obj)


def custom_json_dumps(data):
    return json.dumps(data, default=serialize_object, ensure_ascii=False, indent=4)


# FastAPI App 생성
app = FastAPI(
    title="CrewAI Content Generation API",
    version="1.0",
    description="토픽을 기반으로 CrewAI를 사용하여 콘텐츠를 생성하는 API입니다.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 입력 데이터 모델 정의
class TopicInput(BaseModel):
    topic: str


@app.post("/crewai")
async def crewai_endpoint(input: TopicInput):
    try:
        crew = create_crew()  # Crew 생성
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, crew.kickoff, {"topic": input.topic})
        serialized_result = custom_json_dumps(result)
        return JSONResponse(content=json.loads(serialized_result))
    except Exception as e:
        logger.error(f"CrewAI endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
