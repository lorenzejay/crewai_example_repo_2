from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource


@CrewBase
class CrewaiExampleRepo2Crew:
    """CrewaiExampleRepo2 crew"""

    @agent
    def researcher(self) -> Agent:
        content = "Users name is John. He is 30 years old and lives in San Francisco."
        string_source = StringKnowledgeSource(
            content=content, metadata={"preference": "personal"}
        )
        return Agent(
            config=self.agents_config["researcher"],
            knowledge={"sources": [string_source]},
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
        pdf_path = "/automating_agentic_workflow_generation.pdf"
        pdf_source = PDFKnowledgeSource(
            file_path=[pdf_path],
        )

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            knowledge_sources=[pdf_source],
        )
