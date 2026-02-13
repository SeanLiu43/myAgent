from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
llm = ChatAnthropic(model_name= "claude-sonnet-4-20250514")

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

history = []

while True:
    user_input = input("你: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    response = chain.invoke({"history": history, "input": user_input})
    print(f"AI: {response.content}")

    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response.content))