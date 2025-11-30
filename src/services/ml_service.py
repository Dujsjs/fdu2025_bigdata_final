import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from src.services.ricequant_service import RiceQuantService
from src.core.load_config import settings
import hashlib
from dataclasses import dataclass

@dataclass
class MLPackConfig:
    """
    MLPack的配置信息，由用户指定
    """
    start_date: str
    end_date: str
    selected_instruments: List[str]

@dataclass
class MLPackAddresses:
    """
    MLPack中涉及的各种文件地址（绝对地址），MLPack初始化时自动生成
    """
    raw_data_addr: str = ""
    features_data_addr: str = ""
    value_analysis_model_addr: str = ""
    risk_analysis_model_addr: str = ""
    return_prediction_model_addr: str = ""

class MLPack:
    """
    MLPack将配置、模型等信息打包到一起，每个类型的合约对应一个实例，
    相当于总共最多有5个实例 (CS, ETF, INDX, Future, Option)
    """
    def __init__(self, contract_type: str, config: MLPackConfig):
        """
        初始化MLPack实例

        Args:
            contract_type: 合约类型 ('CS', 'ETF', 'INDX', 'Future', 'Option')
            config: MLPack配置信息
        """
        # 配置和数据路径信息
        self.contract_type = contract_type
        self.config = config
        self.addr = MLPackAddresses()

        # 训练好的模型
        self.trained_models = {
            'value_analysis_model': None,
            'risk_analysis_model': None,
            'return_prediction_model': None
        }
        if self.addr.value_analysis_model_addr and self.addr.risk_analysis_model_addr and self.addr.return_prediction_model_addr:
            self._load_models()

        # 数据
        self.raw_data = None
        self.features_data = None
        if self.addr.raw_data_addr:
            self.raw_data = pd.read_csv(self.addr.raw_data_addr)
        if self.addr.features_data_addr:
            self.features_data = pd.read_csv(self.addr.features_data_addr)

        print('MLPack 预加载完成')

    def _train_value_analysis_model(self) -> Any:
        """
        训练价值分析模型
        """
        pass

    def _train_risk_analysis_model(self) -> Any:
        """
        训练风险分析模型
        """
        pass

    def _train_return_prediction_model(self) -> Any:
        """
        训练收益预测模型
        """
        pass

    def _update_models(self, force_retrain: bool = False) -> None:
        """
        更新模型（若用户指定的合约与训练使用的合约数据不同，
        则与用户对话，当用户确认需重新训练时调用该函数）

        Args:
            force_retrain: 是否强制重新训练
        """
        pass

    def _load_models(self) -> None:
        """
        加载已训练的模型
        """
        pass

    def _update_pack(self, new_config: MLPackConfig) -> None:
        """
        更新整个pack的配置和数据

        Args:
            new_config: 新的配置信息
        """
        pass

    def do_analysis(self) -> Dict[str, Any]:
        """
        执行完整的分析流程

        Returns:
            分析结果字典
        """
        pass

    def save_pack(self, save_path: str) -> None:
        """
        保存整个pack到指定路径

        Args:
            save_path: 保存路径
        """
        pass

    @classmethod
    def load_pack(cls, load_path: str) -> 'MLPack':
        """
        从指定路径加载pack

        Args:
            load_path: 加载路径

        Returns:
            MLPack实例
        """
        pass


class MLService:
    """
    MLService 囊括价值分析模型、风险分析模型、收益预测模型，能调用投资建议引擎分析模型结果
    """
    def __init__(self):
        self.ricequant_service = RiceQuantService()
        self.project_path = settings.project.project_dir
        self.models_path = os.path.join(self.project_path, settings.paths.ml_models)
        self.features_data_path = os.path.join(self.project_path, settings.paths.processed_data)
        print("初始化 MLService 成功！")

    def _construct_contract_features(
            self,
            contract_type: str,
            order_book_id: [str],
            start_date: str,
            end_date: str,
    ) -> str:
        """
        构建适用于多种合约类型的全面特征集，不涉及聚合操作
        :param order_book_id: 用户指定的合约代码列表，仅对此部分样本开展特征工程
        :param contract_type: 合约类型 ('CS', 'ETF', 'INDX', 'Future', 'Option')
        :param start_date: 数据的起始日期
        :param end_date: 数据的终止日期
        :return: 包含所有特征的DataFrame的存储地址
        """
        df_addr, df_fields = self.ricequant_service.instruments_features_fetching(contract_type, int(start_date), int(end_date))
        df = pd.read_csv(df_addr)
        order_book_id_str = None
        if order_book_id:
            order_book_id_str = ','.join(sorted(order_book_id))
        order_book_id_hash = hashlib.md5(order_book_id_str.encode('utf-8')).hexdigest()[:10]
        output_path = os.path.join(self.features_data_path, f"{start_date}_{end_date}_{order_book_id_hash}_{contract_type}_features_data.csv")
        if os.path.exists(output_path):
            print("特征文件已存在！")
            return output_path
        else:
            print("特征文件不存在，开始生成")

        # 1. 基础数据验证并选择合适的样本&按时间排序
        if df.empty:
            raise ValueError("输入数据为空")
        if order_book_id and 'order_book_id' in df.columns:     # 筛选出order_book_id在给定列表中的行
            df = df[df['order_book_id'].isin(order_book_id)]
        if 'date' in df.columns:
            df = df.sort_values(['date', 'order_book_id'])   # 整体数据优先【按照时间排序】

        # 2. 标准化列名（处理可能的大小写差异）
        df = df.copy()
        df.columns = [col.lower() for col in df.columns]

        # 3. 按合约类型构造特征
        if contract_type not in ['CS', 'ETF', 'INDX', 'Future', 'Option']:
            raise ValueError(f"不支持的合约类型: {contract_type}. 必须是 CS, ETF, INDX, Future, Option")

        # 4. 初始化特征DataFrame
        features = pd.DataFrame(index=df.index)
        features['date'] = df['date']
        features['order_book_id'] = df['order_book_id']
        features['close'] = df['close']

        # 关键步骤：创建分组对象
        grouped = df.groupby('order_book_id')

        """ ===== 共享基础特征 (所有合约类型) ===== """
        # 价格特征
        features['returns'] = grouped['close'].transform(lambda x: x.pct_change())
        features['log_returns'] = grouped['close'].transform(lambda x: np.log(x / x.shift(1)))

        df['returns'] = features['returns']     # 无需重新创建 grouped，因为 df 已经更新，grouped 会在访问时使用 df 的最新列
        df['log_returns'] = features['log_returns']

        # 波动率特征
        features['vol_10d'] = grouped['returns'].transform(lambda x: x.rolling(10).std()) * np.sqrt(252)
        features['vol_20d'] = grouped['returns'].transform(lambda x: x.rolling(20).std()) * np.sqrt(252)
        features['vol_60d'] = grouped['returns'].transform(lambda x: x.rolling(60).std()) * np.sqrt(252)
        features['vol_ratio_20_60'] = features['vol_20d'] / features['vol_60d']  # 波动率斜率

        # 趋势特征
        features['ma_5d'] = grouped['close'].transform(lambda x: x / x.rolling(5).mean() - 1)
        features['ma_20d'] = grouped['close'].transform(lambda x: x / x.rolling(20).mean() - 1)
        features['ma_60d'] = grouped['close'].transform(lambda x: x / x.rolling(60).mean() - 1)

        # 动量特征
        df['ma_20d'] = features['ma_20d']
        features['ma_momentum'] = grouped['ma_20d'].transform(lambda x: x - x.shift(5))

        # 真实波幅特征
        if 'high' in df.columns and 'low' in df.columns and 'prev_close' in df.columns:
            # 真实波幅计算
            def calculate_true_range(group):
                prev_close_shifted = group['prev_close'].shift(1)
                true_range_val = np.maximum(
                    group['high'] - group['low'],
                    np.maximum(
                        abs(group['high'] - prev_close_shifted),
                        abs(group['low'] - prev_close_shifted)
                    )
                )
                # 使用前一日收盘价计算百分比TR，注意分母也需要 shift(1)
                return true_range_val / group['prev_close'].shift(1)

            features['true_range'] = grouped.apply(calculate_true_range, include_groups=False).reset_index(level=0, drop=True)
            # ATR
            df['true_range'] = features['true_range']
            features['atr_14d'] = grouped['true_range'].transform(lambda x: x.rolling(14).mean())

        """ ===== 按合约类型添加特定特征 ===== """
        if contract_type in ['CS', 'ETF']:
            """ ===== 股票/ETF 特有特征 ===== """
            # 量能特征
            if 'volume' in df.columns:
                # 滚动均值
                features['volume_10d_ma'] = grouped['volume'].transform(lambda x: x.rolling(10).mean())
                features['volume_ratio'] = df['volume'] / features['volume_10d_ma']
                # 动量
                df['volume_ratio'] = features['volume_ratio']
                features['volume_momentum'] = grouped['volume_ratio'].transform(lambda x: x - x.shift(5))

            if 'total_turnover' in df.columns:
                # 换手率与均值比
                features['turnover_ratio'] = grouped['total_turnover'].transform(lambda x: x / x.rolling(30).mean())

            # 交易活跃度特征
            if 'num_trades' in df.columns:
                features['trade_frequency'] = df['num_trades'] / df['volume']
                # 20日均值
                df['trade_frequency'] = features['trade_frequency']
                features['trade_frequency_20d_ma'] = grouped['trade_frequency'].transform(lambda x: x.rolling(20).mean())
                features['trade_frequency_ratio'] = features['trade_frequency'] / features['trade_frequency_20d_ma']

            # 市场状态特征
            if all(col in df.columns for col in ['close', 'limit_up', 'limit_down']):
                features['is_limit_up'] = (df['close'] >= df['limit_up'] * 0.995).astype(int)
                features['is_limit_down'] = (df['close'] <= df['limit_down'] * 1.005).astype(int)
                # 20日计数
                df['is_limit_up'] = features['is_limit_up']
                df['is_limit_down'] = features['is_limit_down']
                features['limit_up_count_20d'] = grouped['is_limit_up'].transform(lambda x: x.rolling(20).sum())
                features['limit_down_count_20d'] = grouped['is_limit_down'].transform(lambda x: x.rolling(20).sum())

            # 换手率特征（股票特有）：此处涉及外部数据，分组处理难度大，保持原逻辑但需注意外部数据对齐
            features['turnover_rate_approx'] = df['total_turnover'] / (df['close'] * df['volume'])
            df['turnover_rate_approx'] = features['turnover_rate_approx']

        elif contract_type == 'INDX':
            """ ===== 指数特有特征 ===== """
            # 市场广度指标
            if 'high' in df.columns and 'low' in df.columns:
                # 指数波动范围
                features['index_range'] = grouped[['high', 'low', 'close']].apply(
                    lambda x: (x['high'] - x['low']) / x['close'].shift(1),
                    include_groups=False
                ).reset_index(level=0, drop=True)
                # 20日均值
                df['index_range'] = features['index_range']
                features['index_range_20d_ma'] = grouped['index_range'].transform(lambda x: x.rolling(20).mean())

            # 指数动量强度
            features['index_momentum_strength'] = features['returns'] / features['vol_20d']

        elif contract_type in ['Future', 'Option']:
            """ ===== 期货/期权特有特征 ===== """
            # 持仓量特征（期货/期权）
            if 'open_interest' in df.columns:
                # 1日/5日变化
                features['oi_1d_change'] = grouped['open_interest'].transform(lambda x: x.pct_change())
                features['oi_5d_change'] = grouped['open_interest'].transform(lambda x: x.pct_change(5))
                # 动量
                df['oi_1d_change'] = features['oi_1d_change']
                features['oi_momentum'] = grouped['oi_1d_change'].transform(lambda x: x - x.rolling(5).mean())

            settlement_col = 'settlement' if 'settlement' in df.columns else 'close'
            features['settlement'] = df[settlement_col]

            # 期货特有特征：基差和期限结构涉及多个合约的数据对齐，此处保持原逻辑？？？？

        elif contract_type == 'Option':
            # 行权价相关特征
            if 'strike_price' in df.columns:
                features['moneyness'] = df['close'] / df['strike_price']
                # 20日均值
                df['moneyness'] = features['moneyness']
                features['moneyness_20d_ma'] = grouped['moneyness'].transform(lambda x: x.rolling(20).mean())
                features['moneyness_deviation'] = features['moneyness'] - features['moneyness_20d_ma']
            # 隐含波动率估算（简化版）
            if 'strike_price' in df.columns and 'settlement' in df.columns:
                time_to_expiry = 30
                # 隐含波动率的计算不涉及滚动或 shift，但使用 apply 确保在组内操作
                features['implied_vol'] = grouped[['settlement', 'strike_price']].apply(
                    lambda x: np.sqrt(2 * np.pi / time_to_expiry) * (x['settlement'] / x['strike_price']),
                    include_groups=False
                ).reset_index(level=0, drop=True)

        """ ===== 所有合约类型通用的高级特征 ===== """
        # 风险调整收益
        # 夏普比率
        features['sharpe_20d'] = grouped['returns'].transform(lambda x: x.rolling(20).mean()) / features[
            'vol_20d'] * np.sqrt(252)
        df['sharpe_20d'] = features['sharpe_20d']

        # 波动率状态 (qcut 是全局操作，无需分组计算)
        features['vol_regime'] = pd.qcut(features['vol_20d'], q=5, labels=False, duplicates='drop') / 4
        df['vol_regime'] = features['vol_regime']

        # 趋势强度
        trend_window = 20
        # 滚动标准差和均值
        price_std = grouped['close'].transform(lambda x: x.rolling(trend_window).std())
        price_mean = grouped['close'].transform(lambda x: x.rolling(trend_window).mean())
        features['trend_strength'] = (df['close'] - price_mean) / (price_std + 1e-10)
        df['trend_strength'] = features['trend_strength']

        # 尾部风险指标
        # VaR
        features['var_95'] = grouped['returns'].transform(lambda x: x.rolling(60).quantile(0.05))
        df['var_95'] = features['var_95']

        # CVaR(条件风险价值)
        df['cvar_returns_filtered'] = features['returns'].where(features['returns'] <= features['var_95'])
        features['cvar_95'] = grouped['cvar_returns_filtered'].transform(lambda x: x.rolling(60, min_periods=1).mean())   # 在每个合约分组内，对过滤后的（稀疏）收益率计算滚动平均。
        df.drop(columns=['cvar_returns_filtered'], inplace=True)

        # 市场状态综合指标 (基于已分组计算的特征，无需再分组)
        features['market_regime'] = (
            0.4 * features['vol_regime'] +
            0.3 * abs(features['trend_strength']) +
            0.3 * (1 - features['sharpe_20d'].clip(lower=0, upper=1))
        )

        """ ===== 特征工程后处理 ===== """
        MAX_ROLLING_WINDOW = settings.financial_data.features_max_rolling_window
        features = features.groupby('order_book_id').apply(
            lambda x: x.iloc[MAX_ROLLING_WINDOW:, :],
            include_groups=False
        ).reset_index(level=0, drop=False)      # 按 order_book_id 分组，丢弃每个分组的前 MAX_ROLLING_WINDOW 行
        features = features.reset_index(drop=True)
        features = features.replace([np.inf, -np.inf], np.nan)

        # 填充必须在分组后进行，以避免使用下一只股票的数据填充前一只股票的NaN
        features_grouped_for_fillna = features.groupby('order_book_id')
        features = features_grouped_for_fillna.apply(
            lambda x: x.fillna(method='ffill'), include_groups=False).reset_index(level=0, drop=False)   # 不可使用bfill，避免未来信息泄露
        features = features.fillna(0)
        features = features.reset_index(drop=True)

        # # 确保所有特征在合理范围内 (全局统计操作，保持不变)
        # for col in features.columns:
        #     if features[col].dtype in [np.float64, np.float32]:
        #         mean = features[col].mean()
        #         std = features[col].std()
        #         lower_bound = mean - 5 * std
        #         upper_bound = mean + 5 * std
        #         features[col] = features[col].clip(lower=lower_bound, upper=upper_bound)

        # 移除可能由 apply 引入的额外索引
        features = features.sort_values(['date', 'order_book_id'])
        features.to_csv(output_path, index=False)
        return output_path


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
    print(ml_service._construct_contract_features('CS', cs_list, '20240401', '20251128'))