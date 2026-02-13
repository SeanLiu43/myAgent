from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent

# ========== 1. 定义工具 ==========
@tool
def search(query: str) -> str:
    """搜索互联网上的信息。当用户询问你不知道的实时信息时使用。"""
    return f"搜索结果：关于'{query}'的最新信息是..."

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。当用户需要做数学运算时使用。"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"

# ========== 2. 创建 Agent（就这一行！）==========
llm = ChatAnthropic(model_name="claude-sonnet-4-20250514")
agent = create_agent(llm, [search, calculator])

# ========== 3. 对话循环 ==========
history = []

print("LangGraph Agent 已启动（输入 quit 退出）\n")

while True:
    user_input = input("你: ")
    if user_input.lower() in ["quit", "exit"]:
        break

    history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": history})

    # 获取最后一条 AI 回复
    ai_message = result["messages"][-1]
    print(f"AI: {ai_message.content}\n")

    # 更新历史（用完整的消息列表，包含工具调用过程）
    history = result["messages"]

# LangChain = 链式调用，流程是线性的（A → B → C），你提前定义好执行顺序。
# LangGraph = 图状态机，流程可以循环、分支、条件跳转，更适合复杂的 Agent。
# 自动循环的图
# 用户输入 → LLM → 需要工具？ → 是 → 执行工具 → 回到 LLM（再判断）
#                               → 否 → 输出结果