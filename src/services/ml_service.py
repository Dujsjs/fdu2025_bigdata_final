import random
from typing import Dict, Any


class MLService:
    """
    MLService 负责处理金融分析和预测的业务逻辑。
    目前为 Mock 版本，只返回模拟数据。
    """

    def __init__(self):
        # 实际项目中，这里会加载训练好的 ML 模型
        print("初始化 MLService 成功")

    def predict(self, amount: float, risk_level: str) -> Dict[str, Any]:
        """
        根据用户输入的结构化参数，返回模拟的投资分析结果。

        Args:
            amount: 投资金额 (e.g., 50000.0)
            risk_level: 风险等级 (e.g., "low", "medium", "high" 或对应的中文)

        Returns:
            包含预测结果的字典。
        """

        # 定义不同风险等级的模拟年化收益率范围 (包含中文和英文映射)
        risk_map = {
            "low": (0.02, 0.05),
            "保守": (0.02, 0.05),
            "medium": (0.06, 0.10),
            "稳健": (0.06, 0.10),
            "high": (0.12, 0.25),
            "激进": (0.12, 0.25),
        }

        # 统一将输入转为小写以匹配映射表
        level = risk_level.lower()

        # 处理可能的中文输入
        if "保守" in level or "low" in level:
            level_key = "保守"
        elif "稳健" in level or "medium" in level:
            level_key = "稳健"
        elif "激进" in level or "high" in level:
            level_key = "激进"
        else:
            return {
                "error": "风险等级输入无效",
                "message": "请使用 '保守', '稳健', 或 '激进' (或对应的英文)。",
                "amount": amount
            }

        min_rate, max_rate = risk_map[level_key]

        # 模拟生成一个随机年化收益率
        annual_rate = random.uniform(min_rate, max_rate)

        # 计算一年后的预测收益
        predicted_return = amount * annual_rate

        return {
            "risk_level_used": level_key,
            "investment_amount": amount,
            "annual_return_rate": round(annual_rate, 4),
            "predicted_one_year_return": round(predicted_return, 2),
            "disclaimer": "此数据为Mock模型生成的模拟预测结果，不构成真实投资建议。"
        }