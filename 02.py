import os
from dotenv import load_dotenv
from crewai_tools import SerperDevTool
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from crewai import Agent, Task, Crew, Process
# from IPython.display import Markdown, display

load_dotenv()

# llm = ChatOpenAI(
#     model="gemma:2b",
#     base_url="http://localhost:11434/v1"
# )


llm = ChatOllama(
    model="llama3.1",
    base_url="http://localhost:11434")


search_tool = SerperDevTool()


# Define your agents with roles and goals
researcher = Agent(
  role='体育数据分析师',
  goal='发掘巴黎奥运会金牌得主最多的运动员和他们背后的故事',
  backstory="""你是一名经验丰富的体育数据分析师，
  擅长分析和解释各种体育赛事的关键数据和历史记录。
  你能够将复杂的数据转化为有趣且有意义的故事。""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool],
  llm=llm
)

writer = Agent(
  role='体育内容策划师',
  goal='撰写引人入胜的故事，聚焦巴黎奥运会的顶级金牌得主',
  backstory="""你是一位专业的体育内容策划师，以擅长讲述运动员奋斗历程和胜利故事而闻名。
  你能够将数据和事实转化为引人入胜的叙述，激励读者。""",
  verbose=True,
  allow_delegation=False,
  llm=llm
)


# Create tasks for your agents
task1 = Task(
  description="""对巴黎奥运会金牌得主进行全面分析。
  识别获得金牌最多的运动员，并探讨他们的运动生涯和取得成功的关键因素。""",
  expected_output="以要点形式呈现的巴黎奥运会金牌得主分析报告",
  agent=researcher
)

task2 = Task(
  description="""根据提供的金牌得主分析，撰写一篇引人入胜的故事文章，
  重点介绍这些顶级运动员的奋斗历程和背后的故事。
  文章应激励人心，适合广大体育爱好者的读者。""",
  expected_output="至少四段的完整故事文章，务必使用中文",
  agent=writer
)


# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=True,  # Changed from 2 to True
)

# Get your crew to work!
result = crew.kickoff()
print(result)

# display(Markdown(str(result)))