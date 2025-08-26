# script3.py - Stateful agent using ConversationSummaryMemory for holding a simple conversation with user

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain.chains import LLMChain

llm = OllamaLLM(model="gemma2")

prompt = PromptTemplate(
    input_variables=["summary", "input"],
    template='''You are a conversational chatbot. Use the given conversation summary when responding to inputs.
    
        Summary: {summary}

        Input: {input}

    '''
)

memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="summary"
)

agent = LLMChain(
    llm = llm,
    memory = memory,
    prompt = prompt
)

res1 = agent.run(input="hi im valentina")

print(res1)

res2 = agent.run(input="what did i tell you my name was again?")

print(res2)