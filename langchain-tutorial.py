import openai
import langchain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.memory import ConversationBufferMemory

# llm = OpenAI(openai_api_key="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", temperature=0.3)
llm = OpenAI(openai_api_key="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", temperature=0.3)

def practice(): 
    states = [
        "Maharashtra",
        "Gujrat",
        "Punjab"
    ]
    prompt = PromptTemplate.from_template("What is capital of {place}?")
    chain = LLMChain(llm = llm, prompt=prompt)

    for state in states:
        output = chain.run(state)
        print(output)

def simpleLLMChainPractice():
    prompt1 = PromptTemplate.from_template("What is the name of the e-commerce store that sells {product}?")
    chain1 = LLMChain(llm = llm, prompt= prompt1)
    product = "iPhone"
    # output1 = chain.run(product)
    # print(output1)

    prompt2 = PromptTemplate.from_template("What is the name of the e-commerce store that sells {store}?")
    chain2 = LLMChain(llm=llm, prompt=prompt2)
    # store = "Amazon"
    # output2 = chain2.run(store)
    # print(output2)

    # Creating overall chain with chain 1 and chain 2
    chain = SimpleSequentialChain(
        chains=[chain1, chain2], verbose=True
    )

    output = chain.run("candles")
    print(output)

def sequentialChainPractice():
    # This is an LLMChain to write a synopsis given a title of a play.
    template1 = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.

    Title: {title}
    Playwright: This is a synopsis for the above play:"""
    prompt_template1 = PromptTemplate(input_variables=["title"], template=template1)
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template1)


    # This is an LLMChain to write a review of a play given a synopsis.
    template2 = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

    Play Synopsis:
    {synopsis}
    Review from a New York Times play critic of the above play:"""
    prompt_template2 = PromptTemplate(input_variables=["synopsis"], template=template2)
    review_chain = LLMChain(llm=llm, prompt=prompt_template2)

    # This is the overall chain where we run these two chains in sequence.
    overall_chain = SimpleSequentialChain(chains=[synopsis_chain, review_chain], verbose=True)

    review = overall_chain.run("Tragedy at sunset on the beach")
    print(review)
    
    
# Agent Demo
def agentToolPractice():
    tools = load_tools(["wikipedia", "llm-math"], llm=llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    output = agent.run("How old is Rohit Sharma in 2023?")
    print(output)
    

# Memory
prompt1 = PromptTemplate.from_template("What is the name of the e-commerce store that sells {fruit}?")
chain1 = LLMChain(llm = llm, prompt= prompt1, memory=ConversationBufferMemory())
output = chain1.run("fruits")
print(chain1.memory.buffer)
print(output)

