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
  goal='探索大模型的最新发展趋势及其潜在影响',
  backstory="""你是一名经验丰富的人工智能趋势分析师，
  专注于研究大模型技术的进展和应用前景。
  你能够深入分析行业趋势并生成详尽的技术报告。""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool],
  llm=llm
)

writer = Agent(
  role='科技内容创作专家',
  goal='将技术报告转化为引人入胜的文章，适合更广泛的读者',
  backstory="""你是一位擅长将技术性内容转化为有趣和可读性的内容创作专家。
  你的目标是撰写吸引读者的文章，使他们理解大模型技术的影响。""",
  verbose=True,
  allow_delegation=False,
  llm=llm
)


# Create tasks for your agents
task1 = Task(
  description="""对大模型的最新发展和趋势进行深入分析。
  识别关键技术进展、行业影响以及未来的演变方向。""",
  expected_output="大模型发展趋势分析报告，以要点形式呈现",
  agent=researcher
)

task2 = Task(
  description="""基于提供的分析报告，撰写一篇引人入胜的文章，
  使技术趋势内容通俗易懂，并吸引对科技发展感兴趣的读者。
  文章应简明扼要，但富有洞察力和吸引力。""",
  expected_output="完整的四段文章，内容丰富，务必使用中文",
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