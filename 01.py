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
  role='高级研究分析师',
  goal='发掘人工智能和数据科学领域的前沿发展',
  backstory="""你在一家领先的科技智库工作。
  你的专长在于识别新兴趋势。
  你有解析复杂数据并提供可行见解的独特能力。""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool],
  llm=llm
)

writer = Agent(
  role='科技内容策划师',
  goal='撰写关于科技进展的引人入胜的内容',
  backstory="""你是一位知名的内容策划师，以富有洞察力和引人入胜的文章而闻名。
  你能将复杂的概念转化为引人入胜的叙述。""",
  verbose=True,
  allow_delegation=False,
  llm=llm
)


# Create tasks for your agents
# Create tasks for your agents
task1 = Task(
  description="""对2024年最新的AI进展进行全面分析。
  识别关键趋势、突破性技术及其潜在的行业影响。""",
  expected_output="以要点形式呈现的完整分析报告",
  agent=researcher
)

task2 = Task(
  description="""根据提供的见解，撰写一篇引人入胜的博客文章，
  重点介绍最重要的AI进展。
  文章应信息丰富且易于理解，适合技术精通的读者。
  要让文章听起来很酷，避免使用复杂的词汇，以免听起来像是AI写的。""",
  expected_output="至少四段的完整博客文章，务必使用中文",
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