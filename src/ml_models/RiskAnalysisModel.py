from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


class ContractRiskAnalysisModel:
    """
    通用合约风险分析模型（支持5类合约）

    特点：
    1. 根据合约类型自动选择风险特征
    2. 适配各类合约的特有风险模式
    3. 输出风险评分（0-100，越高风险越大）和风险类型
    """

    def __init__(self):
        # 合约类型特定参数
        self.contract_params = {
            'CS': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 20,
                'random_state': 42,
                'n_jobs': -1
            },
            'ETF': {
                'n_estimators': 120,
                'max_depth': 8,
                'min_samples_split': 15,
                'random_state': 42,
                'n_jobs': -1
            },
            'INDX': {
                'n_estimators': 80,
                'max_depth': 12,
                'min_samples_split': 25,
                'random_state': 42,
                'n_jobs': -1
            },
            'Future': {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 30,
                'random_state': 42,
                'n_jobs': -1
            },
            'Option': {
                'n_estimators': 150,
                'max_depth': 12,
                'min_samples_split': 20,
                'random_state': 42,
                'n_jobs': -1
            }
        }

        # 合约类型特定风险范围
        self.risk_ranges = {
            'CS': (0.02, 0.30),  # 股票风险范围（最大回撤）
            'ETF': (0.015, 0.25),  # ETF风险范围
            'INDX': (0.018, 0.28),  # 指数风险范围
            'Future': (0.03, 0.35),  # 期货风险范围
            'Option': (0.04, 0.40)  # 期权风险范围
        }

        # 风险阈值（不同合约类型）
        self.risk_thresholds = {
            'CS': {'low': 30, 'medium': 60, 'high': 85},
            'ETF': {'low': 25, 'medium': 55, 'high': 80},
            'INDX': {'low': 35, 'medium': 65, 'high': 90},
            'Future': {'low': 40, 'medium': 70, 'high': 90},
            'Option': {'low': 45, 'medium': 75, 'high': 95}
        }

        self.model = None
        self.is_trained = False
        self.contract_type = None
        self.risk_features = None

    def train(self, X, y, contract_type, cv_folds=3):
        """
        训练风险分析模型

        参数:
        X: 特征矩阵
        y: 目标变量（未来20日最大回撤的绝对值）
        contract_type: 合约类型
        cv_folds: 交叉验证折数

        返回:
        模型性能指标
        """
        # 1. 验证合约类型
        contract_type = contract_type.upper()
        if contract_type not in self.contract_params:
            raise ValueError(f"不支持的合约类型: {contract_type}")

        # 2. 记录合约类型和特征
        self.contract_type = contract_type
        self.risk_features = X.columns.tolist()

        # 3. 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_results = {
            'train_rmse': [],
            'test_rmse': [],
            'train_r2': [],
            'test_r2': []
        }

        # 4. 交叉验证训练
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # 训练模型
            model = RandomForestRegressor(**self.contract_params[contract_type])
            model.fit(X_train, y_train)

            # 评估
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            cv_results['train_rmse'].append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
            cv_results['test_rmse'].append(np.sqrt(mean_squared_error(y_test, y_test_pred)))
            cv_results['train_r2'].append(r2_score(y_train, y_train_pred))
            cv_results['test_r2'].append(r2_score(y_test, y_test_pred))

        # 5. 使用全部数据重新训练
        self.model = RandomForestRegressor(**self.contract_params[contract_type])
        self.model.fit(X, y)

        # 6. 记录模型状态
        self.is_trained = True

        # 7. 保存模型性能
        performance = {
            'train_rmse': np.mean(cv_results['train_rmse']),
            'test_rmse': np.mean(cv_results['test_rmse']),
            'train_r2': np.mean(cv_results['train_r2']),
            'test_r2': np.mean(cv_results['test_r2']),
            'sample_size': len(X),
            'contract_type': contract_type
        }

        return performance

    def predict_risk_score(self, features):
        """
        预测风险评分（0-100，越高风险越大）

        参数:
        features: 单条特征数据（Series）

        返回:
        float: 风险评分（0-100）
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train方法")

        # 1. 准备预测数据
        pred_data = features[self.risk_features].values.reshape(1, -1)

        # 2. 预测最大回撤
        predicted_max_drawdown = self.model.predict(pred_data)[0]

        # 3. 转换为0-100分（基于合约类型特定范围）
        min_risk, max_risk = self.risk_ranges[self.contract_type]

        # 4. 映射到0-100分
        score = 100 * (predicted_max_drawdown - min_risk) / (max_risk - min_risk)
        return max(0, min(100, score))

    def get_risk_level(self, risk_score):
        """获取风险等级"""
        thresholds = self.risk_thresholds[self.contract_type]
        if risk_score < thresholds['low']:
            return "low"
        elif risk_score < thresholds['medium']:
            return "medium"
        else:
            return "high"

    def get_risk_rationale(self, features, risk_score):
        """
        生成风险评分的理由说明

        参数:
        features: 单条特征数据
        risk_score: 风险评分

        返回:
        list: 理由列表
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")

        # 1. 计算各特征对风险的贡献
        contributions = []

        # 2. 通用风险特征解释
        if 'vol_20d' in features:
            vol_risk = min(max(features['vol_20d'] * 150, 0), 40)
            contributions.append({
                'feature': 'vol_20d',
                'contribution': vol_risk,
                'explanation': f"年化波动率{features['vol_20d'] * 100:.1f}%，增加投资风险"
            })

        if 'cvar_95' in features:
            tail_risk = min(max(features['cvar_95'] * -500, 0), 30)
            contributions.append({
                'feature': 'cvar_95',
                'contribution': tail_risk,
                'explanation': f"尾部风险{features['cvar_95'] * 100:.1f}%，显示极端下跌风险"
            })

        if 'market_regime' in features:
            market_risk = min(max(features['market_regime'] * 100, 0), 25)
            contributions.append({
                'feature': 'market_regime',
                'contribution': market_risk,
                'explanation': f"市场状态风险水平: {['低', '中低', '中', '中高', '高'][int(features['market_regime'] * 4)]}"
            })

        # 3. 合约类型特定风险解释
        if self.contract_type == 'CS':  # 股票
            if 'limit_down_count_20d' in features and features['limit_down_count_20d'] > 1:
                contributions.append({
                    'feature': 'limit_down_count_20d',
                    'contribution': features['limit_down_count_20d'] * 15,
                    'explanation': f"20日内{int(features['limit_down_count_20d'])}次跌停，显示高极端风险"
                })

        elif self.contract_type == 'ETF':  # ETF
            if 'iopv_premium' in features and features['iopv_premium'] > 0.02:
                contributions.append({
                    'feature': 'iopv_premium',
                    'contribution': features['iopv_premium'] * 500,
                    'explanation': f"IOPV溢价率{features['iopv_premium'] * 100:.1f}%，显示估值偏高风险"
                })

        elif self.contract_type == 'INDX':  # 指数
            if 'index_momentum_strength' in features and features['index_momentum_strength'] < -0.8:
                contributions.append({
                    'feature': 'index_momentum_strength',
                    'contribution': (1 + features['index_momentum_strength']) * 50,
                    'explanation': "指数动量强度极低，显示市场弱势风险"
                })

        elif self.contract_type == 'Future':  # 期货
            if 'basis_ratio' in features and features['basis_ratio'] < -0.01:
                contributions.append({
                    'feature': 'basis_ratio',
                    'contribution': abs(features['basis_ratio']) * 1000,
                    'explanation': f"基差比率{features['basis_ratio'] * 100:.2f}%，显示期货深度贴水风险"
                })

            if 'curve_slope' in features and features['curve_slope'] < -0.02:
                contributions.append({
                    'feature': 'curve_slope',
                    'contribution': abs(features['curve_slope']) * 1500,
                    'explanation': f"期限结构斜率{features['curve_slope']:.4f}，显示远期深度贴水风险"
                })

        elif self.contract_type == 'Option':  # 期权
            if 'implied_vol' in features and features['implied_vol'] > 0.4:
                contributions.append({
                    'feature': 'implied_vol',
                    'contribution': (features['implied_vol'] - 0.2) * 200,
                    'explanation': f"隐含波动率{features['implied_vol'] * 100:.1f}%，显示期权价格高估风险"
                })

            if 'vol_skew' in features and features['vol_skew'] > 0.1:
                contributions.append({
                    'feature': 'vol_skew',
                    'contribution': features['vol_skew'] * 300,
                    'explanation': f"波动率偏斜{features['vol_skew']:.3f}，显示市场恐慌情绪风险"
                })

        # 4. 按贡献度排序
        contributions.sort(key=lambda x: x['contribution'], reverse=True)
        return contributions[:3]  # 返回最重要的3条理由

    def calculate_dynamic_position_size(self, risk_score):
        """
        根据风险评分计算动态仓位（0-25%）
        """
        # 风险越高，仓位越低
        max_position = 0.25 * (1 - risk_score / 100)
        # 但不低于5%
        return max(0.05, min(0.25, max_position))

    def calculate_dynamic_stop_loss(self, features, risk_score):
        """
        计算动态止损位（基于ATR和风险评分）
        """
        atr = features.get('atr_14d', 0.02)
        # 风险越高，止损越紧
        stop_loss_multiplier = 1.5 + (risk_score / 100) * 2.5
        return 1 - atr * stop_loss_multiplier