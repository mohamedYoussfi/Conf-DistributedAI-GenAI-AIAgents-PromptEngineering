import os
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools
from crewai import Agent, Task, Crew
from crewai.tasks.task_output import TaskOutput
from crewai.project import crew, agent, task, CrewBase
from langchain_groq import ChatGroq


@CrewBase
class MedicalCrew:
    """Equipage médical"""

    agents_config = "config/medical/medical_agents_config.yaml"

    def __init__(self) -> None:
        self.gpt4_llm = ChatOpenAI(model="gpt-4", temperature=0.4)
        self.groq_llm = ChatGroq(temperature=0.4, model_name="mixtral-8x7b-32768")
        self.human_tools = load_tools(["human"])

    @agent
    def doctor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["agents"]["doctor_agent"],
            llm=self.groq_llm,
        )

    @agent
    def reporter_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["agents"]["reporter_agent"],
            llm=self.groq_llm,
        )

    @task
    def interview_task(self) -> Task:
        return Task(
            config=self.agents_config["tasks"]["interview_task"],
            tools=self.human_tools,
            agent=self.doctor_agent(),
        )

    @task
    def reporter_task(self) -> Task:
        return Task(
            config=self.agents_config["tasks"]["reporter_task"],
            agent=self.reporter_agent(),
        )

    @crew
    def medical_crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, verbose=2)


medical_crew = MedicalCrew()
crew = medical_crew.medical_crew()
result = crew.kickoff()
print(result)
