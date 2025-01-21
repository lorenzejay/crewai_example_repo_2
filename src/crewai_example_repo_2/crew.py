from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource


@CrewBase
class CrewaiExampleRepo2Crew:
    """CrewaiExampleRepo2 crew"""

    @agent
    def researcher(self) -> Agent:
        json_source = JSONKnowledgeSource(file_paths=["user.json"])
        return Agent(
            config=self.agents_config["researcher"],
            knowledge_sources=[json_source],
            verbose=True,
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CrewaiExampleRepo2 crew"""

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
