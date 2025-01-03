
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.playground import Playground, serve_playground_app
from dotenv import load_dotenv
import phi.api
import os

load_dotenv()
phi.api = os.getenv("PHI_API_KEY")

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

app = Playground(agents=[websearch_agent, finance_agent]).get_app()

if __name__ == '__main__':
    serve_playground_app("playground:app", reload=True)