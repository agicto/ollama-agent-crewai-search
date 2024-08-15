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
  role='AI技术分析师',
  goal='探索并总结OpenAI最近的技术更新与发展',
  backstory="""你是一名资深的AI技术分析师，
  专注于研究人工智能领域的最新进展，特别是OpenAI的技术更新。
  你能够深入分析这些更新的技术细节并提供有价值的见解。""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool],
  llm=llm
)

writer = Agent(
  role='技术报告撰写专家',
  goal='撰写详细的技术报告，涵盖OpenAI最近的技术更新及其影响',
  backstory="""你是一位专业的技术报告撰写专家，擅长将复杂的技术内容转化为清晰易懂的报告。
  你的目标是将OpenAI最近的技术更新详细记录并分析其对行业的影响。""",
  verbose=True,
  allow_delegation=False,
  llm=llm
)


# Create tasks for your agents
task1 = Task(
  description="""对OpenAI最近的技术更新进行全面分析。
  识别关键更新、技术进展及其对人工智能领域的潜在影响。""",
  expected_output="包含详细技术分析和行业影响的报告",
  agent=researcher
)

task2 = Task(
  description="""根据提供的分析报告，撰写一篇详细的技术报告，
  重点介绍OpenAI最近的技术更新及其在人工智能领域的意义和影响。
  文章应具有技术深度，适合科技从业者和研究者阅读。""",
  expected_output="完整的四段技术报告，内容详实，务必使用中文",
  agent=writer
)


# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=True,
)

# Get your crew to work!
result = crew.kickoff()
print(result)

# display(Markdown(str(result)))