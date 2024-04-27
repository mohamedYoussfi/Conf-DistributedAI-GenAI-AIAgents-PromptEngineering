from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, task, crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# GROQ_API_KEY = "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"


@CrewBase
class FinancialAnalystCrew:
    """Financial Analyst Crew"""

    agents_config = "config/finance/agents.yaml"

    def __init__(self) -> None:
        self.groq_llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

    @agent
    def company_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["agents"]["company_researcher"], llm=self.groq_llm
        )

    @agent
    def company_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["agents"]["company_analyst"], llm=self.groq_llm
        )

    @task
    def research_company_task(self) -> Task:
        return Task(
            config=self.agents_config["tasks"]["research_company_task"],
            agent=self.company_researcher(),
        )

    @task
    def analyse_company_task(self) -> Task:
        return Task(
            config=self.agents_config["tasks"]["analyse_company_task"],
            agent=self.company_analyst(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents, tasks=self.tasks, process=Process.sequential, verbose=2
        )


def run():
    inputs = {"company_name": "Tesla"}
    result = FinancialAnalystCrew().crew().kickoff(inputs=inputs)
    print("#############")
    print(result)


if __name__ == "__main__":
    run()
