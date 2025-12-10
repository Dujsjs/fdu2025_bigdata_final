import json
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from src.services.rag_service import RAGService
from src.services.ml_service import MLService
from src.services.ricequant_service import RiceQuantService
from src.services.ml_pack import MLPack, MLPackConfig
from typing import List

# 惰性实例化（在需要时才创建服务实例）
_rag_service = None
_ml_service = None
_ricequant_service = None


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

def get_ricequant_service():
    global _ricequant_service
    if _ricequant_service is None:
        _ricequant_service = RiceQuantService()
    return _ricequant_service


# --- 1. RAG知识检索类工具 ---
def get_rag_tool():
    """将 RAGService 提供的查询引擎包装成 LlamaIndex 工具"""

    # 在这里调用服务实例化函数
    rag_service = get_rag_service()
    query_engine = rag_service.get_query_engine()
    return QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="investment_rules_knowledge_base",
            description=(
                "用于查询具体的投资产品规则、产品说明书、法律法规、风险等级定义等文本信息。 "
                "当用户询问'什么是XX规则'或'XX产品有什么限制'时使用。回答来自内部文档。"
            )
        )
    )


# --- 2. 机器学习分析类工具 ---
def get_contract_analysis_tool():
    """将 MLService 的预测功能包装成 LlamaIndex 函数工具"""
    def contract_analysis(contract_type:str, selected_contracts_id:List[str], start_date:str, end_date:str, shibor_type:str, predict_days:int):
        missing_params = []
        if not contract_type:
            missing_params.append("contract_type")
        if not selected_contracts_id or len(selected_contracts_id) == 0:
            missing_params.append("selected_contracts_id")
        if not start_date:
            missing_params.append("start_date")
        if not end_date:
            missing_params.append("end_date")
        if not shibor_type:
            missing_params.append("shibor_type")
        if not predict_days:
            missing_params.append("predict_days")
        if missing_params:
            return json.dumps({
                'error': '参数缺失',
                'missing_parameters': missing_params,
                'message': f'以下参数缺失或为空: {", ".join(missing_params)}，请提供完整参数后再调用工具！'
            }, ensure_ascii=False, indent=2)

        config = MLPackConfig(
            contract_type=contract_type,
            selected_contracts_id=selected_contracts_id,
            start_date=start_date,
            end_date=end_date,
            shibor_type=shibor_type,
            predict_days=predict_days
        )
        pack = MLPack.load_or_build_pack(config)
        report = pack.do_analysis()
        return json.dumps({'分析报告': report}, ensure_ascii=False, indent=2)

    return FunctionTool.from_defaults(
        fn=contract_analysis,
        name="contract_analysis_tool",
        description=(
            "用于对金融合约进行价值分析和预测。当用户需要对合约进行分析时，必须调用此工具。"
            "**重要说明1：在调用此工具前，必须向用户索取并确认以下全部 6 个参数的精确值，在有参数未知时请勿调用工具，而应立即提问！**"
            "**重要说明2：参数较多，可以依次向用户提问！**"
            
            "参数说明："
            "contract_type: 合约类型，包括CS（股票）、ETF（交易所交易基金）、INDX（指数）、Future（期货）、Option（期权）；"
            "selected_contracts_id: 选定的合约代码字符串列表，形如['000001.XSHE','000002.XSHE']"
            "start_date: 分析开始日期（yyyymmdd格式）；"
            "end_date: 分析结束日期（yyyymmdd格式）；"
            "shibor_type: 基准Shibor利率类型（可选值为：ON（隔夜）、1W（1周）、2W（2周）、1M（1个月）、3M（3个月周）、6M（6个月周）、9M（9个月周）、1M（1年））"
            "predict_days: 表示希望预测未来多少日的超额收益率"
            "工具会自动判断是否需要重新训练模型，如果配置发生变化则会重新构建模型。"
        )
    )


# --- 3. 金融信息查询工具 ---
def get_instruments_info_tool():
    """将 RiceQuantService 的查询合约功能包装成 LlamaIndex 函数工具"""
    ricequant_service = get_ricequant_service()
    return FunctionTool.from_defaults(
        fn=ricequant_service.query_stock_info,
        name="instruments_info_tool",
        description=(
            "用于查询当前可交易的金融合约的基本信息。当用户需要查询金融合约相关信息时，必须调用此工具。"
            "所需参数："
            "type: 合约类型，包括CS（股票）、ETF（交易所交易基金）、INDX（指数）、Future（期货）、Option（期权）"
            "query_by_code: 根据合约代码查询"
            "query_by_symbol: 根据合约名称查询"
        )
    )

def get_all_rag_tools():
    """返回所有可用的工具列表"""
    return [get_rag_tool(), get_instruments_info_tool()]