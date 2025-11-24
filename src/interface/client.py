from src.agent.chat_engine import create_invest_agent
from src.services.llm_service import init_llm_and_embed_models
from src.agent.tools import get_rag_service
import asyncio


async def main_async():
    # 1. 初始化 LLM 和 Embedding 模型 (保持不变)
    try:
        init_llm_and_embed_models()
    except Exception as e:
        print(f"致命错误：LLM 或 Embedding 模型初始化失败。请检查 models 配置。错误: {e}")
        return

    # 2. 强制初始化 RAGService (保持不变)
    print("初始化 RAG 知识库...")
    try:
        rag_service = get_rag_service()
        rag_service.get_query_engine()
    except Exception as e:
        print(f"致命错误：RAG 索引加载或构建失败。请检查 data/raw 和 data/storage 目录。错误: {e}")
        return

    # 3. 创建 Agent
    invest_agent = create_invest_agent()

    print("\n--- AI 投资顾问启动成功 ---")
    print("输入 'exit' 或 'quit' 退出。")

    # 4. 进入循环聊天
    while True:
        try:
            user_input = input("\n👤 您: ")
            if user_input.lower() in ["quit", "exit"]:
                print("感谢使用，再见！")
                break

            print("思考中... (Agent 正在推理和调度工具)")
            response = await invest_agent.run(user_input)
            final_answer = response.response
            print(f"\n🤖 顾问: {final_answer}")

        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            if "out of memory" in str(e).lower():
                break


# 定义同步 main 函数作为入口点，启动异步事件循环
def main():
    # 启动异步事件循环
    asyncio.run(main_async())


if __name__ == "__main__":
    main()