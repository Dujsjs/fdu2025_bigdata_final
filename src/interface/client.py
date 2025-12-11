from src.agent.chat_engine import create_invest_rag_agent
from src.services.llm_service import init_llm_and_embed_models
from src.agent.tools import get_rag_service
import asyncio
from src.services.intent_recognition import intent_recognition
from src.services.params_fetching import get_param_CSanalysis, get_param_ETFanalysis, get_param_INDXanalysis, get_param_FUTUREanalysis
from src.services.ml_service import MLService

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

    # 3. 初始化 MLService
    print("初始化机器学习服务...")
    try:
        ml_service = MLService()
    except Exception as e:
        print(f"致命错误：机器学习服务初始化失败。错误: {e}")

    # 4. 进入循环聊天
    print("\n--- AI 投资顾问启动成功 ---")
    print("输入 'exit' 或 'quit' 退出。")
    print("您可以在下方的输入框中简要描述您的需求\n目前支持的金融产品有：股票、基金、指数、期货、期权\n支持的功能有：金融建模与收益预测、投资价值分析、金融知识查询")

    while True:
        try:
            user_input = input("\n👤 您: ")
            if user_input.lower() in ["quit", "exit"]:
                print("感谢使用，再见！")
                break

            print("思考中...(正在推理和调度工具)")
            intent_rst = intent_recognition(user_input)
            print(f"任务列表：{intent_rst}")
            final_answer = ''
            for job in intent_rst:
                job_type = job['job_type']
                user_query = job['user_query']
                additional_info = job['additional_info']
                if job_type == 'rag':
                    invest_agent = create_invest_rag_agent()
                    response = await invest_agent.run(user_query)
                    final_answer += str(response.response)+'\n'+'\n'
                elif job_type == 'ml':
                    if user_query == 'CS':
                        cs_params = get_param_CSanalysis()
                        cs_analysis = str(ml_service.summarize_CSanalysis(start_date=cs_params['start_date'],
                                                        end_date=cs_params['end_date'],
                                                        target_stock_id=cs_params['target_stock_id'],
                                                        order_book_id_list=cs_params['order_book_id_list']))
                        final_answer += cs_analysis+'\n'+'\n'
                    elif user_query == 'ETF':
                        etf_params = get_param_ETFanalysis()
                        etf_analysis = str(ml_service.summarize_ETFanalysis(start_date=etf_params['start_date'],
                                                        end_date=etf_params['end_date'],
                                                        target_ETF_id=etf_params['target_ETF_id'],
                                                        order_book_id_list=etf_params['ETF_id_list']))
                        final_answer += etf_analysis+'\n'+'\n'
                    elif user_query == 'INDX':
                        index_params = get_param_INDXanalysis()
                        index_analysis = str(ml_service.summarize_INDXanalysis(start_date=index_params['start_date'],
                                                        end_date=index_params['end_date'],
                                                        target_index_id=index_params['target_index_id'],
                                                        index_id_list=index_params['index_id_list']))
                        final_answer += index_analysis + '\n' + '\n'
                    elif user_query == 'Future':
                        future_params = get_param_FUTUREanalysis()
                        future_analysis = str(ml_service.summarize_Futureanalysis(start_date=future_params['start_date'],
                                                        end_date=future_params['end_date'],
                                                        target_future_id=future_params['target_future_id'],
                                                        future_id_list=future_params['future_id_list']))
                        final_answer += future_analysis + '\n' + '\n'
                    # elif user_query == 'Option':   # 量化API接口暂无权限，无法分析
                    #     pass

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
