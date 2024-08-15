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
  role='人工智能趋势分析师',
  goal='探索和总结当前大模型排行榜的关键趋势',
  backstory="""你是一名经验丰富的人工智能趋势分析师，
  专注于识别和解读大模型领域的最新动态。
  你擅长将复杂的技术趋势转化为清晰的分析报告。""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool],
  llm=llm
)

writer = Agent(
  role='科技报告撰写专家',
  goal='撰写清晰易懂的大模型排行榜趋势报告',
  backstory="""你是一位资深的科技报告撰写专家，擅长将技术性数据和趋势转化为有洞察力的内容。
  你能够编写详细且易于理解的趋势分析报告，帮助读者掌握行业前沿动态。""",
  verbose=True,
  allow_delegation=False,
  llm=llm
)


# Create tasks for your agents
task1 = Task(
  description="""对当前的大模型排行榜进行全面分析。
  识别其中的关键趋势、主导模型及其在各行业中的影响。""",
  expected_output="以要点形式呈现的大模型排行榜趋势分析报告",
  agent=researcher
)

task2 = Task(
  description="""根据提供的大模型排行榜分析，撰写一篇清晰易懂的趋势报告，
  重点介绍主要趋势和领先的大模型。
  文章应信息丰富，适合对人工智能领域感兴趣的读者。""",
  expected_output="至少四段的完整趋势报告，务必使用中文",
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