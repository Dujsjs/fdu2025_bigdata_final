# 意图识别模块：将用户的输入进行意图识别，并返回相应的json结果，示例如下：
#     {
#         job_type: 'rag'   # 任务类型
#         user_query: '常见的股票投资策略有哪些？它出自知识库的哪一部分？'   # 用户输入：如果是rag就从输入中提取用户的查询语言，
#         additional_info: 'xxx'  # 用户query中额外提供的信息
#     }
#     {
#         job_type: 'ml'   # 任务类型
#         user_query: 'CS'   # 用户输入：如果是ml就从输入中提取金融产品类型（CS-股票、ETF-交易所交易基金、INDX-指数、Future-期货、Option-期权）
#         additional_info: 'xxx'  # 用户query中额外提供的信息
#     }

import ast
import re
from typing import List, Dict
from src.services.llm_service import init_llm_and_embed_models
from llama_index.core import Settings
from src.core.load_config import settings
from llama_index.core.llms import ChatMessage, MessageRole

def intent_recognition(origin_query: str) -> List[Dict[str, str]]:
    """
    通过调用LLM API识别用户查询的意图，返回JSON格式的意图列表。

    参数:
        user_query (str): 用户输入的自然语言查询字符串。

    返回:
        list: 包含一个或多个字典的列表，每个字典有 job_type, user_query, additional_info。
    """
    # 定义提示词模板（已修正格式问题）
    prompt = """你是一个专业的金融领域意图识别系统。请严格按以下规则处理用户查询：
    
    # 任务规则
    1. **意图分类**（job_type）：
       - 选 `rag`：当查询包含知识查询类型的需求时（关键词：有哪些/是什么/策略/出自/查询/解释/定义）
       - 选 `ml`：当查询包含建模分析或建议类型的需求时（关键词：分析/建模/模型/预测/训练/评价/建议）
    
    2. **内容提取**：
       - `rag` 任务：
         • user_query：提取核心问题，如果有多个问题时整合在一起
         • additional_info：提取后续相关补充（如"出自哪部分"类描述）
       - `ml` 任务：
         • user_query：必须映射为标准产品类型（见下表）
         • additional_info：提取建模描述（移除产品名称）
    
    3. **金融产品映射表**：
       | 用户表述       | 标准输出 |
       |--------------|---------|
       | 股票、CS       | "CS"  |
       | 交易所基金、ETF | "ETF"  |
       | 指数、INDX     | "INDX"  |
       | 期货、Future   | "Future" |
       | 期权、Option   | "Option" |
    
    4. **多意图处理**：
       - 每个独立任务生成一个JSON对象
       - 金融产品类型重复出现时整合为单个条目
       - 无额外信息时additional_info=None
    
    # 输出要求
    - 仅输出JSON列表，无任何解释
    - ml类型的user_query必须使用英文名称，不准使用中文
    - 严格使用双引号
    - 禁止添加额外字段
    - 保持原始问题表述不变
    
    # 处理示例
    用户查询："常见的股票投资策略有哪些？它出自知识库的哪一部分？"
    输出：[{"job_type": "rag", "user_query": "常见的股票投资策略有哪些？它出自知识库的哪一部分？", "additional_info": None}]
    
    用户查询："分析CS的波动模型并预测年化收益率，同时预测ETF趋势"
    输出：[
      {"job_type": "ml", "user_query": "CS", "additional_info": "分析波动模型并预测年化收益率"]},
      {"job_type": "ml", "user_query": "ETF", "additional_info": "预测趋势"}
    ]
    
    用户查询："Future期权策略是什么？请用ml建模"
    输出：[
      {"job_type": "rag", "user_query": "Future期权策略是什么？", "additional_info": None},
      {"job_type": "ml", "user_query": "Option", "additional_info": "建模"}
    ]
    
    # 待处理查询："""

    # 调用LLM API
    # 导入 ChatMessage，用于构造对话历史
    messages = [
        ChatMessage(role=MessageRole.USER, content=prompt+str(origin_query))
    ]
    response = Settings.llm.chat(messages)
    llm_response = response.message.content

    # 解析LLM响应并确保返回有效的JSON列表
    MAXLOOP = settings.intent_recognition.max_loop
    n = 0
    while n <= MAXLOOP:
        rst = parse_llm_response(llm_response)
        if len(rst) == 0:
            print('意图识别解析失败，重新尝试中...')
            response = Settings.llm.chat(messages)
            llm_response = response.message.content
            n += 1
        else:
            print('意图识别成功！')
            return rst
    print('超过最大尝试次数！')
    return rst

def parse_llm_response(llm_response: str) -> List[Dict[str, str]]:
    """
    解析LLM的响应，确保返回有效的JSON列表。
    """
    try:   # 先尝试直接解析
        return ast.literal_eval(llm_response)
    except:
        pattern = r'```json\s*([\s\S]*?)```'
        match = re.search(pattern, llm_response, re.IGNORECASE)

        if match:
            content = match.group(1).strip()
            try:
                rst = ast.literal_eval(content)
                return validate_result(rst)
            except:
                pass
    return []


def validate_result(result: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    验证结果是否符合要求的格式。

    参数:
        result (list): 待验证的结果列表。

    返回:
        list: 验证通过的结果列表。
    """
    validated = []
    for item in result:
        # 检查必需字段
        if not isinstance(item, dict):
            continue

        if "job_type" not in item or "user_query" not in item or "additional_info" not in item:
            continue

        # 验证job_type值
        if item["job_type"] not in ["rag", "ml"]:
            continue

        # 验证ml任务的user_query
        if item["job_type"] == "ml":
            valid_products = ["CS", "ETF", "INDX", "Future", "Option"]
            mapping_dict = {"股票": "CS", "交易所基金": "ETF", "指数": "INDX", "期货": "Future", "期权": "Option"}
            if item["user_query"] not in valid_products:
                if item["user_query"] in mapping_dict:
                    item["user_query"] = mapping_dict[item["user_query"]]
                else:
                    continue

        # 确保additional_info是字符串
        if not isinstance(item["additional_info"], str):
            item["additional_info"] = str(item["additional_info"])

        validated.append(item)

    # 如果验证后没有有效结果，返回默认值
    if not validated:
        return []

    return validated

if __name__ == "__main__":
    init_llm_and_embed_models()  # 初始化 LLM 和 Embedding 模型
    query1 = "我想预测一下股票、期货的走势，并分别分析一下它们的投资价值。另外，告诉我投资的核心注意事项是什么？"
    query2 = "金融领域常见的投资规则有哪些？另外，请给出000001股票的投资建议"
    result1 = intent_recognition(query1)
    print(result1)
    result2 = intent_recognition(query2)
    print(result2)