from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# ========== 1. 定义工具 ==========
@tool
def search(query: str) -> str:
    """搜索互联网上的信息。当用户询问你不知道的实时信息时使用。"""
    # 这里是模拟搜索，实际可以接入搜索API
    return f"搜索结果：关于'{query}'的最新信息是..."

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。当用户需要做数学运算时使用。"""
    try:
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"

# ========== 2. 创建带工具的 LLM ==========
tools = [search, calculator]
llm = ChatAnthropic(model_name="claude-sonnet-4-20250514")
llm_with_tools = llm.bind_tools(tools)

# ========== 3. 构建 Chain ==========
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手，可以使用工具来帮助用户。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm_with_tools

# ========== 4. 工具执行函数 ==========
tool_map = {"search": search, "calculator": calculator}

def process_response(response):
    """如果 AI 要调用工具，就执行工具并返回最终结果"""
    if response.tool_calls:
        print(f"  🔧 AI 正在调用工具...")
        messages = [response]  # 先放入 AI 的回复

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            print(f"  🔧 调用: {tool_name}({tool_args})")

            # 执行工具
            result = tool_map[tool_name].invoke(tool_args)
            print(f"  📎 结果: {result}")

            # 把工具结果包装成 ToolMessage
            from langchain_core.messages import ToolMessage
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        # 把工具结果发回给 LLM，让它生成最终回答
        final_response = llm_with_tools.invoke(messages)
        return final_response.content
    else:
        return response.content

# ========== 5. 对话循环 ==========
history = []

print("聊天机器人已启动（输入 quit 退出）")
print("我可以帮你搜索信息和做数学计算！\n")

while True:
    user_input = input("你: ")
    if user_input.lower() in ["quit", "exit"]:
        break

    response = chain.invoke({"history": history, "input": user_input})
    answer = process_response(response)
    print(f"AI: {answer}\n")

    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=answer))

#  Agent 的核心思想：LLM 作为"大脑"，自己决定用不用工具、用哪个工具、怎么用结果