from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

websearch_agent = Agent(
        name='websearch_agent', 
        role = 'search the web for the information you need',
        model=Groq(id='llama3-groq-70b-8192-tool-use-preview'), 
        tools=[DuckDuckGo()],
        instructions=["Always show the sources used"],
        show_tools_calls = True,
        markdown=True

)

finance_agent = Agent(
    name='finance_agent',
    model=Groq(id='llama-3.1-70b-versatile'),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True),
        ],
        instructions=["Use tables while displaying the data"],
        show_tool_calls=True,
        markdown=True

)
multi_agent = Agent(
    team=[websearch_agent, finance_agent],
    model=Groq(id="llama-3.1-70b-versatile"),
    instructions=["Always inclue sources used, Use tables while displaying the data"],
    markdown=True,
    show_tool_calls=True
)

multi_agent.print_response("Summarize analyst recommendations & show latest news for NVIDIA stock", stream=True)
