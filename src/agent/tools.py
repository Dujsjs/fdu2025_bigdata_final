import json
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from src.services.rag_service import RAGService
from src.services.ml_service import MLService

# 惰性实例化（在需要时才创建服务实例）
_rag_service = None
_ml_service = None


def get_rag_service():
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def get_ml_service():
    global _ml_service
    if _ml_service is None:
        _ml_service = MLService()
    return _ml_service


# --- 1. RAG 知识检索工具 ---

def get_rag_tool():
    """将 RAGService 提供的查询引擎包装成 LlamaIndex 工具"""

    # 在这里调用服务实例化函数
    rag_service = get_rag_service()
    query_engine = rag_service.get_query_engine()

    return QueryEngineTool(
        query_engine=query_engine,
        # ... (metadata 保持不变)
        metadata=ToolMetadata(
            name="investment_rules_knowledge_base",
            description=(
                "用于查询具体的投资产品规则、产品说明书、法律法规、风险等级定义等文本信息。 "
                "当用户询问'什么是XX规则'或'XX产品有什么限制'时使用。回答来自内部文档。"
            )
        )
    )


# --- 2. 机器学习分析工具 ---

def get_ml_tool():
    """将 MLService 的预测功能包装成 LlamaIndex 函数工具"""

    ml_service = get_ml_service()

    def perform_investment_analysis(amount: float, risk_level: str) -> str:
        # 调用 MLService 的预测方法
        result_dict = ml_service.predict(amount, risk_level)
        return json.dumps(result_dict, ensure_ascii=False)

    return FunctionTool.from_defaults(
        fn=perform_investment_analysis,
        name="financial_prediction_tool",
        description=(
            "用于执行基于数据的分析和计算。当用户提供具体的投资金额和风险偏好，"
            "并要求进行'预测'、'计算'或'分析收益'时，必须调用此工具。"
        )
    )


def get_all_tools():
    """返回所有可用的工具列表"""
    return [get_rag_tool(), get_ml_tool()]