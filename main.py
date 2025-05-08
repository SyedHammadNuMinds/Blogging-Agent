import os
import yaml
import json
from typing import List
from pydantic import BaseModel, Field

from crewai.llm import LLM
from crewai import Agent, Task, Crew
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

# Loading the environment variables
load_dotenv()

# Initializing the LLM with proper error handling
try:
    llm = LLM(model=os.environ["GROQ_MODEL_NAME"], api_key=os.environ["GROQ_API_KEY"])
except KeyError as e:
    print(f"Environment variable not found: {e}")
    print("Please make sure GROQ_MODEL_NAME and GROQ_API_KEY are set in your .env file")
    exit(1)

# Loading config files with error handling
files = {"agents": "config/agents.yaml", "tasks": "config/tasks.yaml"}

configs = {}
for config_type, file_path in files.items():
    try:
        with open(file_path, "r") as file:
            configs[config_type] = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Config file not found: {file_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML in {file_path}: {e}")
        exit(1)

agents_config = configs["agents"]
tasks_config = configs["tasks"]


# Pydantic Model for structured output
class OutputModel(BaseModel):
    blog: str = Field(..., description="Detailed Blog")
    seo_keywords: List[str] = Field(..., description="List of SEO keywords")
    citations: List[str] = Field(..., description="List of citations")


# Improved search tool with error handling
@tool("DuckDuckGoSearch")
def search_tool(search_query: str) -> str:
    """
    Search the web for information based on a specific query.

    Args:
        search_query: A focused search query about a specific topic or claim needing citation.
                     Should be concise and specific (not an entire blog post).

    Returns:
        str: Search results that can be used for citations.
    """
    # Ensure the query isn't too long
    if len(search_query) > 500:
        search_query = search_query[:500]  # Truncate if too long

    try:
        results = DuckDuckGoSearchRun().run(search_query)
        # Handle empty results
        if not results or len(results.strip()) == 0:
            return "No search results found. Try a different search query."
        return results
    except Exception as e:
        return f"Search error occurred: {str(e)}. Try a more specific query."


# Agents
task_expansion_agent = Agent(config=agents_config["topic_expansion_agent"], llm=llm)
seo_strategy_agent = Agent(config=agents_config["seo_strategy_agent"], llm=llm)
research_citation_agent = Agent(
    config=agents_config["research_citation_agent"],
    llm=llm,
    tools=[search_tool],  # Directly assign tool to the agent
)

# Tasks
create_blog_task = Task(
    config=tasks_config["create_blog_task"], agent=task_expansion_agent
)

find_seo_keywords_task = Task(
    config=tasks_config["find_seo_keywords_task"],
    agent=seo_strategy_agent,
    context=[create_blog_task],
)

reference_finder_task = Task(
    config=tasks_config["reference_finder_task"],
    agent=research_citation_agent,
    context=[create_blog_task],
)

# Crew with error handling and timeouts
crew = Crew(
    agents=[task_expansion_agent, seo_strategy_agent, research_citation_agent],
    tasks=[create_blog_task, find_seo_keywords_task, reference_finder_task],
    verbose=True,
    process_timeout=600,  # Set a reasonable timeout (10 minutes)
)


def run_crew(topic):
    """Run the crew with error handling"""
    try:
        inputs = {"input_topic": topic}
        result = crew.kickoff(inputs=inputs)
        return result
    except Exception as e:
        print(f"Error running crew: {e}")
        return None


# Example usage
if __name__ == "__main__":
    topic = input("Enter a topic for the blog: ")
    result = run_crew(topic)
    if result:
        print("\n=== FINAL RESULT ===")
        print(json.dumps(result, indent=2))
