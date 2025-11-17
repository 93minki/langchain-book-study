from crewai import Crew, Process
from agents import planner, writer, editor, translator
from tasks import plan, write, edit, translate_task


def create_crew():
    # Crew 생성
    crew = Crew(
        agents=[planner, writer, editor, translator],  # 에이전트 목록
        tasks=[plan, write, edit, translate_task],  # 작업 목록
        process=Process.sequential,
        verbose=True,  # 상세한 로그 출력 여부
    )
    return crew
