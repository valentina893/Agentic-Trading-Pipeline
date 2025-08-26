# script2.py - Stateful agent using ConversationBufferMemory for holding a simple conversation with user

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

llm = OllamaLLM(model="gemma2")

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template='''You are a conversational chatbot. Use the given conversation history when responding to inputs.
    
        History: {history}

        Input: {input}

    '''
)

memory = ConversationBufferMemory(memory_key="history")

agent = LLMChain(
    llm = llm,
    memory = memory,
    prompt = prompt
)

res1 = agent.run(input="hi im valentina")

print(res1)

res2 = agent.run(input="what did i tell you my name was again?")

print(res2)