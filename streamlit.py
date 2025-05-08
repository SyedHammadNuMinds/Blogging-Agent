import os

import yaml
from crewai import Agent, Crew, Task
from crewai.llm import LLM
from crewai.tools import tool
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun

import streamlit as st

# Load environment variables
load_dotenv()

# LLM
llm = LLM(model=os.environ["GROQ_MODEL_NAME"], api_key=os.environ["GROQ_API_KEY"])

# Load YAML configs
with open("config/agents.yaml", "r") as f:
    agents_config = yaml.safe_load(f)
with open("config/tasks.yaml", "r") as f:
    tasks_config = yaml.safe_load(f)

# Streamlit UI
st.title("📝 Blog Generator Agentic App")
input_topic = st.text_input("Enter a blog topic")

if st.button("Generate Blog"):
    if not input_topic:
        st.error("Please enter a topic to proceed.")
    else:
        # Create agents with topic
        topic_expansion_agent = Agent(
            config=agents_config["topic_expansion_agent"], llm=llm, verbose=True
        )
        seo_agent = Agent(
            config=agents_config["seo_strategy_agent"], llm=llm, verbose=True
        )
        research_agent = Agent(
            config=agents_config["research_citation_agent"], llm=llm, verbose=True
        )
        final_agent = Agent(
            config=agents_config["final_formatter_agent"], llm=llm, verbose=True
        )

        # Create tasks with topic
        task1 = Task(
            config=tasks_config["create_blog_task"],
            agent=topic_expansion_agent,
            input_values={"input_topic": input_topic},
        )
        task2 = Task(
            config=tasks_config["find_seo_keywords_task"],
            agent=seo_agent,
            input_values={"input_topic": input_topic},
        )
        task3 = Task(
            config=tasks_config["reference_finder_task"],
            agent=research_agent,
            input_values={"input_topic": input_topic},
        )
        task4 = Task(
            config=tasks_config["finalize_blog_task"],
            agent=final_agent,
            context=[task1, task2, task3],
            input_values={"input_topic": input_topic},
        )

        # Run the crew
        crew = Crew(
            agents=[topic_expansion_agent, seo_agent, research_agent, final_agent],
            tasks=[task1, task2, task3, task4],
            verbose=True,
        )

        inputs = {"input_topic": input_topic}

        result = crew.kickoff(inputs=inputs)

        # Display the result
        st.subheader("🧾 Final Blog Post")
        st.markdown(result)
