from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Settings
from src.agent.tools import get_all_rag_tools

def create_invest_rag_agent():
    """
    创建并返回具备多轮信息收集和工具调用的金融 ReAct Agent。
    """

    if Settings.llm is None:
        raise RuntimeError("LLM 尚未初始化。请确保LLMService已被调用。")

    # 1. 获取所有工具
    tools = get_all_rag_tools()

    # 2. 配置记忆 (用于多轮对话)
    memory = ChatMemoryBuffer.from_defaults(
        # 限制记忆大小，留出上下文空间
        token_limit=int(Settings.llm.context_window * 0.7)
    )

    # 3. 定义 Agent 的系统角色和流程指导 (与 ChatAgent 的 Prompt 兼容)
    FINANCIAL_SYSTEM_PROMPT = (
        "你是一位专业的AI金融投资顾问。你的目标是高效、准确地为用户提供投资咨询和分析。\n"
        "请严格遵循以下流程:\n"
        "1. **信息收集/判断**: \n"
        "   - **预测/分析请求**: 如果用户要求进行收益预测或分析，你必须获取两个关键参数：**投资金额 (amount)** 和 **风险偏好 (risk_level)**。\n"
        "   - **规则/知识请求**: 如果用户仅询问规则或定义，可以直接使用 `investment_rules_knowledge_base`。\n"
        "2. **主动提问**: 如果用户请求需要预测/分析，但缺少金额或风险偏好参数，**你必须主动发起提问**，明确要求用户提供缺失的信息。\n"
        "3. **工具调用**: 当且仅当所有必需参数（金额和风险偏好）收集完整后，你才能调用 `financial_prediction_tool` 进行分析。\n"
        "4. **综合回答**: 综合 RAG 知识和 ML 分析结果，提供专业的、负责任的投资建议。回答时保持专业和简洁的风格。"
    )

    # 4. 构建 ReAct Agent (直接使用 ReActAgent 替代 ChatAgent/AgentRunner 组合)
    agent = ReActAgent(
        tools=tools,
        llm=Settings.llm,
        memory=memory,
        verbose=True,
        system_prompt=FINANCIAL_SYSTEM_PROMPT
    )

    # print("Agent: ReAct Agent (具备多轮对话信息收集能力) 组装完成。")
    return agent