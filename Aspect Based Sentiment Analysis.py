from crewai import Agent, Process, Task, Crew
from crewai.project import agent, task, CrewBase, crew
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import json

OPEN_AI_KEY = "....."
GROQ_API_KEY = "....."


@CrewBase
class SentimentAnalysisCrew:
    """Aspect based sentiment analysis crew"""

    agents_config = "config/sentiment-analysis/agents.yaml"

    def __init__(self) -> None:
        self.groq_llm = ChatGroq(
            model_name="mixtral-8x7b-32768", temperature=0, api_key=GROQ_API_KEY
        )
        self.gpt4_llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=OPEN_AI_KEY)

    @agent
    def sentiment_analysis_agent(self) -> Agent:
        return Agent(
            llm=self.groq_llm,
            config=self.agents_config["agents"]["systiment_analysis_agent"],
        )

    @task
    def setiment_analysis_task(self) -> Task:
        return Task(
            config=self.agents_config["tasks"]["sentiment_analysis_task"],
            agent=self.sentiment_analysis_agent(),
        )

    @crew
    def sentiment_analysis_crew(self) -> Crew:
        return Crew(
            agents=self.agents, tasks=self.tasks, verbose=2, process=Process.sequential
        )


reviews = [
    "The screen is good. The keyboard is bad and the mousepad is quite",
    "The screen is good. The keyboard is bad and the mousepad is good",
    "The screen good. The keyboard is quite and the mousepad is good",
]

crew = SentimentAnalysisCrew().sentiment_analysis_crew()
results = []
for review in reviews:
    result = crew.kickoff(inputs={"review": review})
    print("####################")
    print(result)
    results.append(json.loads(result))

print("========================")
print(results)
