# script1.py - Stateless agent using a custom PromptTemplate for summarizing inputs

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = OllamaLLM(model="gemma2")

prompt = PromptTemplate(
    template="Summarize the given input with one sentence. Input: {input}"
)

agent = LLMChain(
    llm = llm,
    prompt = prompt
)

res = agent.run(input="In a quiet mountain village, residents have recently begun a project to restore an old stone bridge that connects the town square to nearby hiking trails. The bridge, built more than two centuries ago, suffered damage from harsh winters and heavy rains, making it unsafe for frequent use. Volunteers from the community are working together on weekends, using both traditional masonry techniques and modern tools to preserve its original charm while ensuring durability. The effort has also sparked renewed interest in local history, as younger generations learn about the bridge’s role in trade and travel long ago.")

print(res)