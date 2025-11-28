import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from src.services.ricequant_service import RiceQuantService
from src.core.load_config import settings
import hashlib

# class MLPack:   # 将数据、模型、配置（如时间范围、模型参数）封装为一个pack，形成类似“记忆”的功能
#

class MLService:
    """
    MLService 囊括价值分析模型、风险分析模型、收益预测模型，能调用投资建议引擎分析模型结果
    """
    def __init__(self):
        # 实际项目中，这里会加载训练好的 ML 模型
        self.ricequant_service = RiceQuantService()
        self.project_path = settings.project.project_dir
        self.models_path = os.path.join(self.project_path, settings.paths.ml_models)
        self.features_data_path = os.path.join(self.project_path, settings.paths.processed_data)
        print("初始化 MLService 成功！")

    def construct_contract_features(
            self,
            contract_type: str,
            order_book_id: [str],
            start_date: str,
            end_date: str,
            prev_contract_data: Optional[pd.DataFrame] = None
    ) -> str:
        """
        构建适用于多种合约类型的全面特征集
        :param order_book_id: 用户指定的合约代码列表，仅对此部分样本开展特征工程
        :param df: 从get_price获取的原始数据DataFrame
        :param contract_type: 合约类型 ('CS', 'ETF', 'INDX', 'Future', 'Option')
        :param prev_contract_data: 用于期货展期等场景的前序合约数据（可选）
        :return: 包含所有特征的DataFrame的存储地址
        """
        df_addr, df_fields = self.ricequant_service.instruments_features_fetching(contract_type, int(start_date), int(end_date))
        df = pd.read_csv(df_addr)

        # 1. 基础数据验证并选择合适的样本&按时间排序
        if df.empty:
            raise ValueError("输入数据为空")
        if order_book_id and 'order_book_id' in df.columns:     # 筛选出order_book_id在给定列表中的行
            df = df[df['order_book_id'].isin(order_book_id)]
        if 'date' in df.columns:
            df = df.sort_values(['order_book_id', 'date'])

        # 2. 标准化列名（处理可能的大小写差异）
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]

        # 3. 按合约类型构造特征
        if contract_type not in ['CS', 'ETF', 'INDX', 'Future', 'Option']:
            raise ValueError(f"不支持的合约类型: {contract_type}. 必须是 CS, ETF, INDX, Future, Option")

        # 4. 初始化特征DataFrame
        features = pd.DataFrame(index=df.index)

        """ ===== 共享基础特征 (所有合约类型) ===== """
        # 价格特征
        features['close'] = df['close']
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # 波动率特征
        features['vol_10d'] = features['returns'].rolling(10).std() * np.sqrt(252)
        features['vol_20d'] = features['returns'].rolling(20).std() * np.sqrt(252)
        features['vol_60d'] = features['returns'].rolling(60).std() * np.sqrt(252)
        features['vol_ratio_20_60'] = features['vol_20d'] / features['vol_60d']  # 波动率斜率

        # 趋势特征
        features['ma_5d'] = df['close'] / df['close'].rolling(5).mean() - 1
        features['ma_20d'] = df['close'] / df['close'].rolling(20).mean() - 1
        features['ma_60d'] = df['close'] / df['close'].rolling(60).mean() - 1
        features['ma_momentum'] = features['ma_20d'] - features['ma_20d'].shift(5)

        # 真实波幅特征
        if 'high' in df.columns and 'low' in df.columns and 'prev_close' in df.columns:
            true_range = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['prev_close'].shift(1)),
                    abs(df['low'] - df['prev_close'].shift(1))
                )
            )
            features['true_range'] = true_range / df['prev_close'].shift(1)
            features['atr_14d'] = features['true_range'].rolling(14).mean()

        """ ===== 按合约类型添加特定特征 ===== """
        if contract_type in ['CS', 'ETF']:
            """ ===== 股票/ETF 特有特征 ===== """
            # 量能特征
            if 'volume' in df.columns:
                features['volume_10d_ma'] = df['volume'].rolling(10).mean()
                features['volume_ratio'] = df['volume'] / features['volume_10d_ma']
                features['volume_momentum'] = features['volume_ratio'] - features['volume_ratio'].shift(5)

            if 'total_turnover' in df.columns:
                features['turnover_ratio'] = df['total_turnover'] / df['total_turnover'].rolling(30).mean()

            # 交易活跃度特征
            if 'num_trades' in df.columns:
                features['trade_frequency'] = df['num_trades'] / df['volume']
                features['trade_frequency_20d_ma'] = features['trade_frequency'].rolling(20).mean()
                features['trade_frequency_ratio'] = features['trade_frequency'] / features['trade_frequency_20d_ma']

            # 市场状态特征
            if all(col in df.columns for col in ['close', 'limit_up', 'limit_down']):
                features['is_limit_up'] = (df['close'] >= df['limit_up'] * 0.995).astype(int)
                features['is_limit_down'] = (df['close'] <= df['limit_down'] * 1.005).astype(int)
                features['limit_up_count_20d'] = features['is_limit_up'].rolling(20).sum()
                features['limit_down_count_20d'] = features['is_limit_down'].rolling(20).sum()

            # 换手率特征（股票特有）
            if contract_type == 'CS' and 'total_turnover' in df.columns and 'volume' in df.columns:
                # 假设已获取流通股本（需外部数据），这里用近似方法
                if prev_contract_data is not None and 'float_shares' in prev_contract_data.columns:
                    float_shares = prev_contract_data['float_shares'].iloc[-1]
                    features['turnover_rate'] = df['volume'] / float_shares
                else:
                    # 用成交额/价格近似换手率
                    features['turnover_rate_approx'] = df['total_turnover'] / (df['close'] * df['volume'])

        elif contract_type == 'INDX':
            """ ===== 指数特有特征 ===== """
            # 市场广度指标（需成分股数据，这里用近似方法）
            if 'high' in df.columns and 'low' in df.columns:
                features['index_range'] = (df['high'] - df['low']) / df['close'].shift(1)
                features['index_range_20d_ma'] = features['index_range'].rolling(20).mean()

            # 指数动量强度
            features['index_momentum_strength'] = features['returns'] / features['vol_20d']

        elif contract_type in ['Future', 'Option']:
            """ ===== 期货/期权特有特征 ===== """
            # 持仓量特征（期货/期权）
            if 'open_interest' in df.columns:
                features['oi_1d_change'] = df['open_interest'].pct_change()
                features['oi_5d_change'] = df['open_interest'].pct_change(5)
                features['oi_momentum'] = features['oi_1d_change'] - features['oi_1d_change'].rolling(5).mean()

            # 结算价处理（期货）
            settlement_col = 'settlement' if 'settlement' in df.columns else 'close'
            features['settlement'] = df[settlement_col]

            # 期货特有特征
            if contract_type == 'Future':
                # 基差计算（需现货价格，这里假设prev_contract_data包含现货数据）
                if prev_contract_data is not None and 'spot_price' in prev_contract_data.columns:
                    features['basis'] = df[settlement_col] - prev_contract_data['spot_price']
                    features['basis_ratio'] = features['basis'] / prev_contract_data['spot_price']
                    features['basis_momentum'] = features['basis_ratio'] - features['basis_ratio'].shift(5)

                # 期限结构特征（需多个到期合约数据）
                if prev_contract_data is not None and 'next_settlement' in prev_contract_data.columns:
                    features['curve_slope'] = (df[settlement_col] - prev_contract_data['next_settlement']) / df[
                        settlement_col]

            # 期权特有特征
            elif contract_type == 'Option':
                # 行权价相关特征
                if 'strike_price' in df.columns:
                    features['moneyness'] = df['close'] / df['strike_price']
                    features['moneyness_20d_ma'] = features['moneyness'].rolling(20).mean()
                    features['moneyness_deviation'] = features['moneyness'] - features['moneyness_20d_ma']

                # 合约乘数相关
                if 'contract_multiplier' in df.columns:
                    features['contract_value'] = df['close'] * df['contract_multiplier']

                # 隐含波动率估算（简化版）
                if 'strike_price' in df.columns and 'settlement' in df.columns:
                    time_to_expiry = 30  # 假设30天到期
                    features['implied_vol'] = np.sqrt(2 * np.pi / time_to_expiry) * (
                                df['settlement'] / df['strike_price'])

        """ ===== 所有合约类型通用的高级特征 ===== """
        # 风险调整收益
        features['sharpe_20d'] = features['returns'].rolling(20).mean() / features['vol_20d'] * np.sqrt(252)

        # 波动率状态
        features['vol_regime'] = pd.qcut(features['vol_20d'], q=5, labels=False, duplicates='drop') / 4

        # 趋势强度
        trend_window = 20
        price_std = df['close'].rolling(trend_window).std()
        price_mean = df['close'].rolling(trend_window).mean()
        features['trend_strength'] = (df['close'] - price_mean) / (price_std + 1e-10)

        # 尾部风险指标
        features['var_95'] = features['returns'].rolling(60).quantile(0.05)
        features['cvar_95'] = features['returns'][features['returns'] <= features['var_95']].rolling(60).mean()

        # 市场状态综合指标
        features['market_regime'] = (
                0.4 * features['vol_regime'] +
                0.3 * abs(features['trend_strength']) +
                0.3 * (1 - features['sharpe_20d'].clip(lower=0, upper=1))
        )

        """ ===== 特征工程后处理 ===== """
        # 处理无穷大和NaN值
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(method='bfill')

        # 确保所有特征在合理范围内
        for col in features.columns:
            if features[col].dtype in [np.float64, np.float32]:
                # 将极端值限制在5个标准差内
                mean = features[col].mean()
                std = features[col].std()
                lower_bound = mean - 5 * std
                upper_bound = mean + 5 * std
                features[col] = features[col].clip(lower=lower_bound, upper=upper_bound)

        # 添加时间特征（对季节性分析有用）
        if not df.index.empty:
            if isinstance(df.index, pd.DatetimeIndex):
                features['day_of_week'] = df.index.dayofweek
                features['month'] = df.index.month
                features['quarter'] = df.index.quarter

        if order_book_id:
            order_book_id_str = ','.join(sorted(order_book_id))
        order_book_id_hash = hashlib.md5(order_book_id_str.encode('utf-8')).hexdigest()[:10]
        output_path = os.path.join(self.features_data_path, f"{start_date}_{end_date}_{order_book_id_hash}_features_data.csv")
        features.reset_index(inplace=True)
        if 'index' in features.columns:
            features.drop(columns=['index'], inplace=True)
        features.to_csv(output_path, index=False)
        return output_path

    # def

    # def predict(self, amount: float, risk_level: str) -> Dict[str, Any]:
    #     """
    #     根据用户输入的结构化参数，返回模拟的投资分析结果。
    #
    #     Args:
    #         amount: 投资金额 (e.g., 50000.0)
    #         risk_level: 风险等级 (e.g., "low", "medium", "high" 或对应的中文)
    #
    #     Returns:
    #         包含预测结果的字典。
    #     """
    #
    #     # 定义不同风险等级的模拟年化收益率范围 (包含中文和英文映射)
    #     risk_map = {
    #         "low": (0.02, 0.05),
    #         "保守": (0.02, 0.05),
    #         "medium": (0.06, 0.10),
    #         "稳健": (0.06, 0.10),
    #         "high": (0.12, 0.25),
    #         "激进": (0.12, 0.25),
    #     }
    #
    #     # 统一将输入转为小写以匹配映射表
    #     level = risk_level.lower()
    #
    #     # 处理可能的中文输入
    #     if "保守" in level or "low" in level:
    #         level_key = "保守"
    #     elif "稳健" in level or "medium" in level:
    #         level_key = "稳健"
    #     elif "激进" in level or "high" in level:
    #         level_key = "激进"
    #     else:
    #         return {
    #             "error": "风险等级输入无效",
    #             "message": "请使用 '保守', '稳健', 或 '激进' (或对应的英文)。",
    #             "amount": amount
    #         }
    #
    #     min_rate, max_rate = risk_map[level_key]
    #
    #     # 模拟生成一个随机年化收益率
    #     annual_rate = random.uniform(min_rate, max_rate)
    #
    #     # 计算一年后的预测收益
    #     predicted_return = amount * annual_rate
    #
    #     return {
    #         "risk_level_used": level_key,
    #         "investment_amount": amount,
    #         "annual_return_rate": round(annual_rate, 4),
    #         "predicted_one_year_return": round(predicted_return, 2),
    #         "disclaimer": "此数据为Mock模型生成的模拟预测结果，不构成真实投资建议。"
    #     }

if __name__ == '__main__':
    ml_service = MLService()
    cs_list = [
        "000001.XSHE",
        "000002.XSHE",
        "000004.XSHE",
        "000006.XSHE",
        "000007.XSHE",
        "000008.XSHE",
        "000009.XSHE",
        "000010.XSHE",
        "000011.XSHE",
        "000012.XSHE",
        "000014.XSHE",
        "000016.XSHE",
        "000017.XSHE",
        "000019.XSHE",
        "000020.XSHE",
        "000021.XSHE",
        "000025.XSHE",
        "000026.XSHE",
        "000027.XSHE",
        "000028.XSHE",
        "000029.XSHE",
        "000030.XSHE",
        "000031.XSHE",
        "000032.XSHE",
        "000034.XSHE",
        "000035.XSHE",
        "000036.XSHE"
]
    print(ml_service.construct_contract_features('CS', cs_list, '20240401', '20251128'))