# script4.py - Agentic pipeline that gives trading advice on various stocks using DuckDuckGoSearchQuery

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.chains import LLMChain

llm = OllamaLLM(model="gemma2")

duckduckgo = DuckDuckGoSearchResults(max_results=3)

search_query = "Apple stock price and recent news"
web_search = duckduckgo.run(search_query)

print("web search:")
print(web_search)

researcher_prompt = PromptTemplate(
    template='''You are a financial research assistant.

    Given this web search result: {web_search}

    Extract the following:
    1. Current Apple (AAPL) stock price
    2. A summary of major Apple-related news in the past 7 days

    Format your response like this:

    Stock Price: ...
    News Summary: ...

'''
)

researcher_agent = LLMChain(
    llm=llm,
    prompt=researcher_prompt
)

analyst_prompt = PromptTemplate(
    template='''You are a stock analyst.

    Given this data from the researcher:
    {summary}

    Evaluate if it's a good time to buy Apple stock.
    Base your judgment on:
    - Whether the stock is rising or falling
    - Whether recent news is positive or negative

    Then respond in this format:

    Analysis: ...
    Decision: Yes, it's a good time to buy. OR No, it's not a good time.
'''
)

analyst_agent = LLMChain(
    llm=llm,
    prompt=analyst_prompt
)

summary = researcher_agent.run(web_search=web_search)

print("researcher:")
print(summary)

decision = analyst_agent.run(summary=summary)

print("analyst:")
print(decision)
