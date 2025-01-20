from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

# Web search agent

web_search_agent=Agent(
    name="Web search Agent",
    role="Search the web for the information",
    model=Groq(id="llama-3.2-90b-vision-preview"),
    tools=[DuckDuckGo()],
    isnstructions=["Always include the sources"],
    show_tools_call=True,
    markdown=True
)

# Financial Agent
Financial_agent=Agent(
    name="Financial AI Agent",
    model=Groq(id="llama-3.2-90b-vision-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_news=True)],
    instructions=['use tables to display the data'],
    markdown=True
)

multi_ai_agent=Agent(

    team=[web_search_agent,Financial_agent],

    model=Groq(id="llama-3.1-70b-versatile"),

    instructions=["Always include sources", "Use tables to display data"],

    show_tool_calls=True,

    markdown=True,
)

multi_ai_agent.print_response("Summarize analyst recommendations and share the latest news for NVDA", stream=True)