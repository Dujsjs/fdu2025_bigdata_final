import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Dict, List, Union
from src.core.load_config import settings
from src.services.ricequant_service import RiceQuantService
import joblib

class ContractAnalysisModel:
    """
    通用合约价值分析模型（支持5类合约）

    特点：
    1. 根据合约类型自动选择特征
    2. 适配各类合约的特有价值驱动因素
    3. 输出可解释的价值评分（0-100）
    """

    def __init__(self, contract_type: str):
        # 合约类型特定参数
        self.contract_params = settings.mlModels.parameters

        # 合约类型特定价值范围
        self.value_ranges = {
            'CS': (-0.03, 0.08),  # 股票价值范围（年化）
            'ETF': (-0.02, 0.06),  # ETF价值范围
            'INDX': (-0.025, 0.07),  # 指数价值范围
            'Future': (-0.04, 0.10),  # 期货价值范围
            'Option': (-0.05, 0.15)  # 期权价值范围
        }

        self.model = None
        self.model_performace = None
        self.shap_explainer = None
        self.is_trained = False
        self.contract_type = contract_type
        self.value_features = None
        self.predict_days = None
        self.EARLY_STOPPING_ROUND = settings.mlModels.early_stopping_rounds

    def train(self, X, y):
        """
        训练价值分析模型

        参数:
        X: 特征矩阵
        y: 目标变量
        contract_type: 合约类型
        cv_folds: 交叉验证折数

        返回:
        模型性能指标
        """
        # 2. 记录合约类型和特征
        self.value_features = X.columns.tolist()

        # 3. 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=settings.mlModels.cv_fold)
        cv_results = {
            'train_mae': [],
            'test_mae': [],
            'train_r2': [],
            'test_r2': [],
            'best_n_estimators': []
        }

        # 4. 交叉验证训练
        print('开始训练模型')
        for fold, (historical_idx, test_idx) in enumerate(tqdm(tscv.split(X))):
            X_historical, y_historical = X.iloc[historical_idx], y.iloc[historical_idx]

            N_historical = len(X_historical)
            VALID_RATIO = 0.2
            valid_size = int(N_historical * VALID_RATIO)

            # 使用定义的常量 EARLY_STOPPING_ROUNDS
            if valid_size < self.EARLY_STOPPING_ROUND:
                print(f"Warning: Fold {fold + 1}: Validation set size ({valid_size}) is too small. Skipping fold.")
                continue

            train_subset_idx = N_historical - valid_size
            X_train = X_historical.iloc[:train_subset_idx]
            y_train = y_historical.iloc[:train_subset_idx]
            X_valid = X_historical.iloc[train_subset_idx:]
            y_valid = y_historical.iloc[train_subset_idx:]

            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            # 转换为 DMatrix 格式
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalid = xgb.DMatrix(X_valid, label=y_valid)
            dtest = xgb.DMatrix(X_test)

            eval_list = [(dtrain, 'train'), (dvalid, 'validation')]

            # 使用 xgb.train 进行训练，参数兼容性最高
            bst = xgb.train(
                params=self.contract_params,  # 包含 objective, learning_rate, eval_metric 等
                dtrain=dtrain,
                num_boost_round=settings.mlModels.num_boost_rounds,
                evals=eval_list,
                early_stopping_rounds=self.EARLY_STOPPING_ROUND,
                verbose_eval=False
            )

            # 记录最佳迭代次数
            best_n_estimators = bst.best_iteration
            cv_results['best_n_estimators'].append(best_n_estimators)

            # 使用最佳迭代次数对训练集和测试集进行评估
            y_train_pred = bst.predict(dtrain, iteration_range=(0, best_n_estimators))
            y_test_pred = bst.predict(dtest, iteration_range=(0, best_n_estimators))

            cv_results['train_mae'].append(mean_absolute_error(y_train, y_train_pred))
            cv_results['test_mae'].append(mean_absolute_error(y_test, y_test_pred))
            cv_results['train_r2'].append(r2_score(y_train, y_train_pred))
            cv_results['test_r2'].append(r2_score(y_test, y_test_pred))

        # 计算最佳轮数的平均值
        avg_best_n_estimators = int(np.mean(cv_results['best_n_estimators']))
        print(f"交叉验证平均最佳迭代次数: {avg_best_n_estimators}")

        # 4. 使用全部数据重新训练最终模型（使用平均最佳轮数）
        # 最终模型使用 XGBRegressor 封装器，便于后续集成（例如 SHAP）
        print(f'使用全部数据和平均最佳迭代次数 {avg_best_n_estimators} 重新训练最终模型...')
        final_params = self.contract_params.copy()
        final_params['n_estimators'] = avg_best_n_estimators
        final_params.pop('eval_metric', None)  # 最终训练无需监控指标
        self.model = xgb.XGBRegressor(**final_params)
        self.model.fit(X, y)

        # 5. 创建SHAP解释器
        self.shap_explainer = shap.TreeExplainer(self.model)

        # 6. 记录模型状态
        self.is_trained = True

        # 7. 保存模型性能
        performance = {
            'train_mae': np.mean(cv_results['train_mae']),
            'test_mae': np.mean(cv_results['test_mae']),
            'train_r2': np.mean(cv_results['train_r2']),
            'test_r2': np.mean(cv_results['test_r2']),
            'avg_n_estimators': avg_best_n_estimators,  # 增加平均最佳轮数
            'sample_size': len(X)
        }
        self.model_performace = performance

    def predict_excess_return(self, features: pd.Series) -> float:
        """预测未来20日超额收益"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")
        pred_data = features[self.value_features].values.reshape(1, -1)
        return self.model.predict(pred_data)[0]

    def predict_value_score(self, features: pd.Series) -> float:
        """预测投资价值评分（0-100分）"""
        predicted_excess_returns = self.predict_excess_return(features)
        min_value, max_value = self.value_ranges[self.contract_type]

        # 映射到0-100分，使用Sigmoid或Sigmoid-like函数进行平滑，避免简单的线性截断
        # 简单线性映射（保持原样，但进行了截断）
        score = 100 * (predicted_excess_returns - min_value) / (max_value - min_value)
        return max(0, min(100, score))

    def get_value_rationale(self, features: pd.Series) -> List[Dict[str, Union[str, float]]]:
        """生成价值评分的理由说明（基于SHAP值）"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        if self.shap_explainer is None:
            return []

        # 1. 计算SHAP值
        pred_data = features[self.value_features].values.reshape(1, -1)
        # 注意：TreeExplainer.shap_values() 的输出是一个数组或列表
        shap_values = self.shap_explainer.shap_values(pred_data)

        if isinstance(shap_values, list):
            # For multi-output models, though usually just one for regression
            shap_values = shap_values[0]

            # 2. 生成解释
        contributions: List[Dict[str, Union[str, float]]] = []
        for i, feature in enumerate(self.value_features):
            shap_value = shap_values[0][i] if shap_values.ndim == 2 else shap_values[i]

            # 仅关注对预测有显著影响的特征
            if abs(shap_value) < 0.005:
                continue

            explanation = self._get_feature_explanation(feature, shap_value, features)

            contributions.append({
                'feature': feature,
                'shap_value': float(shap_value),
                'explanation': explanation
            })

        # 3. 按绝对SHAP值排序
        contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        return contributions

    def _get_feature_explanation(self, feature: str, shap_value: float, features: pd.Series) -> str:
        """
        生成更专业、更可解释的特征贡献说明
        """
        is_positive = shap_value > 0
        direction = "正向" if is_positive else "负向"

        # 提取值
        value = features.get(feature, np.nan)
        if pd.isna(value):
            return f"特征 {feature} 数据缺失，对价值产生了 {direction} 影响。"

        # --- 通用特征解释 ---
        if feature == 'ma_20d':
            diff_percent = value * 100
            return f"价格相对20日均线乖离度（{diff_percent:.2f}%）影响：乖离度{'高于' if diff_percent > 0 else '低于'}零值，模型视为{direction}信号。"

        elif feature == 'vol_ratio_20_60':
            vol_ratio = value
            analysis = ""
            if vol_ratio > 1.1:
                analysis = "短期波动率显著高于长期，预示市场进入高波动或趋势可能反转。"
            elif vol_ratio < 0.9:
                analysis = "短期波动率低于长期，显示市场情绪趋于稳定。"
            else:
                analysis = "波动率结构平稳。"
            return f"波动率斜率 ({vol_ratio:.2f}) 分析：{analysis}，模型视为{direction}信号。"

        elif feature == 'sharpe_20d':
            return f"历史20日夏普比率 ({value:.3f})：体现近期风险调整收益水平，对价值有{direction}影响。"

        elif feature == 'var_95':
            return f"60日历史VaR(95%) ({value * 100:.2f}%)：体现尾部风险水平，风险越低对价值越有{direction}影响。"

        # --- 合约类型特定解释 ---
        elif self.contract_type == 'CS' and feature == 'turnover_ratio':
            ratio_percent = value * 100
            sentiment = "强势资金流入" if value > 1.5 else ("温和活跃" if value > 1.0 else "交投清淡")
            return f"换手率与均值比 ({ratio_percent:.0f}%)：市场活跃度高，模型捕捉到{sentiment}带来的{direction}影响。"

        elif self.contract_type == 'Future' and feature == 'settlement':
            return f"当前结算价 ({value:.2f}) 对价值预测有{direction}影响。"

        elif self.contract_type == 'Option' and feature == 'implied_vol':
            iv_percent = value * 100
            sentiment = "溢价" if iv_percent > 30 else "低估"
            return f"隐含波动率 ({iv_percent:.1f}%)：IV水平相对{'较高' if is_positive else '较低'}，市场情绪偏{sentiment}，模型视为{direction}信号。"

        # 默认通用解释
        return f"{feature} (当前值: {value:.3f}) 对投资价值有 {direction} 影响。"

    def _analyze_risk_features(self, features: pd.Series) -> Dict[str, Union[str, float]]:
        """
        基于关键风险特征，给出定性风险评估
        """
        risk_level = "中"
        vol_20d = features.get('vol_20d', np.nan)
        var_95 = features.get('var_95', np.nan)
        cvar_95 = features.get('cvar_95', np.nan)

        # 1. 波动率评估
        if not pd.isna(vol_20d):
            if vol_20d > 0.4:
                risk_level = "高"
            elif vol_20d < 0.15:
                if risk_level != "高":  # 不覆盖高风险
                    risk_level = "低"

        # 2. 尾部风险评估 (CVaR的绝对值越大，尾部风险越高)
        if not pd.isna(cvar_95) and cvar_95 < -0.05:
            risk_level = "高"  # 历史最大损失风险大，直接定为高风险

        return {
            "level": risk_level,
            "volatility": vol_20d,
            "var_95": var_95,
            "cvar_95": cvar_95
        }

    def generate_investment_report(self, features: pd.Series) -> str:
        """
        生成包含价值、风险和收益预测的综合投资分析报告 (Markdown 格式)
        """
        # 1. 核心预测
        value_score = self.predict_value_score(features)
        predicted_returns = self.predict_excess_return(features)

        # 2. 风险分析
        risk_data = self._analyze_risk_features(features)

        # 3. 价值理由
        rationale = self.get_value_rationale(features)

        # 4. 报告生成
        report_markdown = []
        report_markdown.append(f"# 合约价值分析报告 - {self.contract_type}")
        report_markdown.append(f"## 🚀 投资价值评分：{value_score:.1f}/100")

        # 价值等级判断
        if value_score >= 80:
            value_grade = "极具吸引力 (Strong Buy)"
        elif value_score >= 60:
            value_grade = "中高价值 (Buy)"
        elif value_score >= 40:
            value_grade = "中性偏多 (Hold)"
        else:
            value_grade = "低价值/高估 (Sell)"

        report_markdown.append(f"**评估结论:** **{value_grade}**")
        report_markdown.append("\n---")

        report_markdown.append("## 📊 投资收益预测")
        report_markdown.append(f"基于模型预测，未来20日超额收益率（年化）预期为: **{predicted_returns * 100:.2f}%**")
        report_markdown.append("\n---")

        report_markdown.append("## 🛡️ 风险分析 (Investment Risk)")
        report_markdown.append(f"**当前风险水平：** **{risk_data['level']}**")

        report_markdown.append("### 风险指标快照：")
        report_markdown.append(f"- **20日波动率 (Volatility):** {risk_data['volatility']:.3f} (反映短期价格震荡程度)")
        report_markdown.append(f"- **VaR 95% (最大亏损):** {risk_data['var_95'] * 100:.2f}% (60日历史数据，95%置信度下的最大亏损)")
        report_markdown.append(
            f"- **CVaR 95% (平均尾部亏损):** {risk_data['cvar_95'] * 100:.2f}% (95%置信度下平均最差损失，**尾部风险关键指标**)")
        report_markdown.append("\n---")

        report_markdown.append("## 💡 价值驱动因素 (SHAP 解释)")
        report_markdown.append("以下是模型预测该价值评分的**主要原因**（按影响力排序）：")

        for item in rationale:
            impact = "（积极贡献）" if item['shap_value'] > 0 else "（消极贡献）"
            report_markdown.append(f"- **{item['feature']}** {impact}: {item['explanation']}")

        report_markdown.append("\n---")
        report_markdown.append(f"模型由 XGBoost 训练，目标变量为未来{self.predict_days}日超额收益。")

        return "\n".join(report_markdown)

    def preprocess_features_data(self, features_data: pd.DataFrame, start_date:str, end_date:str, shibor_type:str, predict_days:int):
        """
        通用数据预处理函数

        Args:
            features_data (pd.DataFrame): 原始特征数据
            start_date (str): 开始日期
            end_date (str): 结束日期
            shibor_type (str): Shibor利率类型，如'1W'
            predict_days (int): 预测天数

        Returns:
            tuple: (X_train, y_train) 处理后的特征和目标变量
        """
        self.predict_days = predict_days

        # 删除所有全零列
        features_data = features_data.loc[:, ~(features_data == 0).all(axis=0)]

        # 复制数据
        data = features_data.copy()
        x_col = data.columns.to_list()

        # 确保按id和日期排序
        features_data_sorted = features_data.sort_values(['order_book_id', 'date'])

        # 计算未来收益（shift的天数与predict_days相关）
        data['future_returns'] = features_data_sorted.groupby('order_book_id')['close'].transform(
            lambda x: x.shift(-predict_days) / x - 1
        )

        # 分组缩尾操作（Winsorization）
        def winsorize_series(series, lower_percentile=0.05, upper_percentile=0.95):
            lower_bound = series.quantile(lower_percentile)
            upper_bound = series.quantile(upper_percentile)
            return series.clip(lower=lower_bound, upper=upper_bound)
        data['future_returns'] = data.groupby('order_book_id')['future_returns'].transform(winsorize_series)

        # 合并特征和目标变量（需要rice_quant_service实例）
        rice_quant_service = RiceQuantService()
        data = rice_quant_service.merge_shibor_data(data, start_date, end_date, [shibor_type], predict_days)
        data['excess_returns'] = data['future_returns'] - data[shibor_type]

        # 删除任何包含NaN的行
        data = data.dropna()

        # 分离X和y
        X_train = data[x_col].set_index(['date', 'order_book_id'])
        y_train = data[['date', 'order_book_id', 'excess_returns']].set_index(['date', 'order_book_id'])

        return X_train, y_train

    def save_model(self, file_path: str):
        """
        保存模型实例到本地文件
        """
        joblib.dump(self, file_path)
        print(f"模型已保存至: {file_path}")

    @classmethod
    def load_model(cls, file_path: str):
        """
        从本地文件加载模型实例，此为类方法，可通过类名称直接调用
        """
        model_instance = joblib.load(file_path)
        return model_instance


if __name__ == '__main__':
    # 1. 初始化模型
    value_model = ContractAnalysisModel('CS')
    features_data = pd.read_csv(r"/root/nas-private/bigdata_final_project/data/processed/20240401_20251128_3d96b3a4bf_CS_features_data.csv")

    # features_data = features_data.loc[:, ~(features_data == 0).all(axis=0)]   # 删除所有全零列
    #
    # # 3. 定义目标变量 y (未来20日超额收益)
    # # 直接在原始数据上操作，保持顺序不变
    # data = features_data.copy()
    # x_col = data.columns.to_list()
    # features_data_sorted = features_data.sort_values(['order_book_id', 'date'])  # 确保按股票和日期排序
    # data['future_returns'] = features_data_sorted.groupby('order_book_id')['close'].transform(lambda x: x.shift(-5) / x - 1)
    #
    # # 分组缩尾操作（Winsorization）
    # def winsorize_series(series, lower_percentile=0.05, upper_percentile=0.95):
    #     lower_bound = series.quantile(lower_percentile)
    #     upper_bound = series.quantile(upper_percentile)
    #     return series.clip(lower=lower_bound, upper=upper_bound)
    # data['future_returns'] = data.groupby('order_book_id')['future_returns'].transform(winsorize_series)
    #
    # # 合并特征和目标变量
    # data = rice_quant_service.merge_shibor_data(data, '20240401', '20251128', ['1W'], 3)
    # data['excess_returns'] = data['future_returns'] - data['1W']
    #
    # # 删除任何包含NaN的行
    # data = data.dropna()
    #
    # # 分离X和y
    # X_train = data[x_col].set_index(['date', 'order_book_id'])
    # y_train = data[['date', 'order_book_id', 'excess_returns']]
    # y_train = y_train.set_index(['date', 'order_book_id'])

    # 4. 训练模型
    X_train, y_train = value_model.preprocess_features_data(features_data, '20240401', '20251128', '1W', 3)
    value_model.train(X_train, y_train)
    print("\n--- 模型训练性能 ---")
    print(value_model.model_performace)
    print("---------------------------------")

    # 4. 预测并生成报告
    latest_features = X_train.iloc[-1].copy()  # 获取最新一行数据 (Series)
    report = value_model.generate_investment_report(latest_features)

    print("\n--- 📝 最新投资分析报告 ---")
    print(report)
    print("-----------------------------")