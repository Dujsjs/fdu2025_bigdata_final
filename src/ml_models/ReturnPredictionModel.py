import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


class LSTMModel(nn.Module):
    """PyTorch LSTM模型（替代TensorFlow实现）"""
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]

        # 应用批归一化（需要转置）
        lstm_out = lstm_out.unsqueeze(1)
        lstm_out = self.batch_norm(lstm_out)
        lstm_out = lstm_out.squeeze(1)

        lstm_out = self.dropout(lstm_out)
        x = self.fc1(lstm_out)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class ContractReturnPredictionModel:
    """
    通用合约收益预测模型（支持5类合约，使用PyTorch实现）

    特点：
    1. 根据合约类型自动选择收益预测特征
    2. 适配各类合约的价格行为模式
    3. 输出上涨概率和置信度
    """

    def __init__(self, sequence_length=20):
        """
        初始化收益预测模型

        参数:
        sequence_length: LSTM序列长度（天）
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.contract_type = None
        self.return_features = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 合约类型特定模型参数
        self.model_params = {
            'CS': {
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'learning_rate': 0.001
            },
            'ETF': {
                'hidden_size': 50,
                'num_layers': 2,
                'dropout': 0.15,
                'learning_rate': 0.001
            },
            'INDX': {
                'hidden_size': 70,
                'num_layers': 2,
                'dropout': 0.25,
                'learning_rate': 0.001
            },
            'Future': {
                'hidden_size': 80,
                'num_layers': 2,
                'dropout': 0.3,
                'learning_rate': 0.001
            },
            'Option': {
                'hidden_size': 100,
                'num_layers': 2,
                'dropout': 0.35,
                'learning_rate': 0.001
            }
        }

    def create_sequences(self, X, y):
        """
        创建LSTM序列

        参数:
        X: 特征DataFrame
        y: 目标变量Series

        返回:
        X_seq, y_seq
        """
        X_seq, y_seq = [], []

        # 确保数据是numpy数组
        X_np = X.values
        y_np = y.values

        # 创建序列
        for i in range(len(X_np) - self.sequence_length):
            X_seq.append(X_np[i:i + self.sequence_length])
            y_seq.append(y_np[i + self.sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def build_model(self, input_size, contract_type):
        """
        构建PyTorch LSTM模型

        参数:
        input_size: 输入特征维度
        contract_type: 合约类型

        返回:
        PyTorch模型
        """
        # 获取合约类型特定参数
        params = self.model_params.get(contract_type.upper(), self.model_params['CS'])

        # 创建模型
        model = LSTMModel(
            input_size=input_size,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        )

        return model.to(self.device)

    def train(self, X, y, contract_type, validation_split=0.2, epochs=50, batch_size=32):
        """
        训练收益预测模型

        参数:
        X: 特征DataFrame
        y: 目标变量Series
        contract_type: 合约类型
        validation_split: 验证集比例
        epochs: 训练轮数
        batch_size: 批量大小

        返回:
        训练历史
        """
        # 1. 记录合约类型和特征
        self.contract_type = contract_type.upper()
        if self.contract_type not in self.model_params:
            self.contract_type = 'CS'
        self.return_features = X.columns.tolist()

        # 2. 特征缩放
        X_scaled = self.scaler.fit_transform(X)

        # 3. 创建序列
        X_seq, y_seq = self.create_sequences(X_scaled, y)

        # 4. 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).reshape(-1, 1).to(self.device)  # 确保形状正确

        # 5. 分割训练集和验证集
        split_idx = int(len(X_tensor) * (1 - validation_split))
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

        # 6. 构建模型
        self.model = self.build_model(X_seq.shape[2], self.contract_type)

        # 7. 定义损失函数和优化器
        criterion = nn.BCELoss()
        params = self.model_params[self.contract_type]
        optimizer = optim.Adam(self.model.parameters(), lr=params['learning_rate'])

        # 8. 训练模型
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            optimizer.zero_grad()

            # 前向传播
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 计算训练准确率
            train_preds = (outputs > 0.5).float()
            train_acc = accuracy_score(y_train.cpu().numpy(), train_preds.cpu().numpy())

            # 验证模式
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val)

                # 计算验证准确率
                val_preds = (val_outputs > 0.5).float()
                val_acc = accuracy_score(y_val.cpu().numpy(), val_preds.cpu().numpy())

            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            train_accs.append(train_acc)
            val_accs.append(val_acc)

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracy': train_accs[-1],
            'val_accuracy': val_accs[-1]
        }

    def predict(self, features_df):
        """
        预测未来5日上涨概率

        参数:
        features_df: 特征DataFrame（至少包含sequence_length行）

        返回:
        dict: 预测结果
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")

        # 1. 准备数据
        features = features_df[self.return_features].tail(self.sequence_length)

        # 2. 特征缩放
        features_scaled = self.scaler.transform(features)

        # 3. 创建序列
        X_seq = np.array([features_scaled])

        # 4. 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        # 5. 预测
        self.model.eval()
        with torch.no_grad():
            probability = self.model(X_tensor).item()

        # 6. 计算置信度（基于预测概率与0.5的距离）
        confidence = 0.5 + abs(probability - 0.5)

        # 7. 确定置信级别
        if confidence > 0.75:
            confidence_level = "high"
        elif confidence > 0.65:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        return {
            'horizon': '5d',
            'direction': 'up' if probability > 0.5 else 'down',
            'probability': float(probability),
            'confidence': float(confidence),
            'confidence_level': confidence_level
        }

    def get_feature_contributions(self, features_df):
        """
        获取特征对预测的贡献（简化版）

        参数:
        features_df: 特征DataFrame

        返回:
        list: 特征贡献列表
        """
        # 简化实现：使用最后几个时间步的特征值作为贡献
        recent_features = features_df[self.return_features].iloc[-1]

        contributions = []
        for feature in self.return_features:
            value = recent_features[feature]

            # 合约类型特定解释
            explanation = self._get_feature_explanation(feature, value)

            contributions.append({
                'feature': feature,
                'value': float(value),
                'explanation': explanation
            })

        # 按绝对值排序
        contributions.sort(key=lambda x: abs(x['value']), reverse=True)
        return contributions[:3]  # 返回最重要的3个特征

    def _get_feature_explanation(self, feature, value):
        """
        生成特征解释（合约类型特定）
        """
        # 通用解释（所有合约类型）
        if feature == 'ma_20d':
            if value > 0.05:
                return f"价格高于20日均线{value * 100:.1f}%，预示上涨"
            elif value < -0.05:
                return f"价格低于20日均线{abs(value) * 100:.1f}%，预示下跌"

        elif feature == 'trend_strength':
            if value > 0.8:
                return "趋势强度极高，显示持续上涨可能性大"
            elif value < -0.8:
                return "趋势强度极低，显示持续下跌风险高"

        # 合约类型特定解释
        if self.contract_type == 'CS':  # 股票
            if feature == 'volume_ratio':
                if value > 1.2:
                    return f"成交量为30日均值{value * 100:.0f}%，显示资金流入"
                elif value < 0.8:
                    return f"成交量为30日均值{value * 100:.0f}%，显示资金流出"

        elif self.contract_type == 'ETF':  # ETF
            if feature == 'iopv_premium':
                if value > 0.01:
                    return f"IOPV溢价率{value * 100:.1f}%，显示估值偏高"
                elif value < -0.01:
                    return f"IOPV溢价率{value * 100:.1f}%，显示估值偏低"

        elif self.contract_type == 'INDX':  # 指数
            if feature == 'index_momentum_strength':
                if value > 0.8:
                    return "指数动量强度极高，显示市场强势"
                elif value < -0.8:
                    return "指数动量强度极低，显示市场弱势"

        elif self.contract_type == 'Future':  # 期货
            if feature == 'basis_ratio':
                if value > 0.005:
                    return f"基差比率{value * 100:.2f}%，显示期货溢价"
                elif value < -0.005:
                    return f"基差比率{value * 100:.2f}%，显示期货贴水"

            elif feature == 'curve_slope':
                if value > 0.01:
                    return f"期限结构斜率{value:.4f}，显示远期升水"
                elif value < -0.01:
                    return f"期限结构斜率{value:.4f}，显示远期贴水"

        elif self.contract_type == 'Option':  # 期权
            if feature == 'implied_vol':
                if value > 0.3:
                    return f"隐含波动率{value * 100:.1f}%，显示期权价格偏高"
                elif value < 0.2:
                    return f"隐含波动率{value * 100:.1f}%，显示期权价格偏低"

            elif feature == 'vol_skew':
                if value > 0.05:
                    return f"波动率偏斜{value:.3f}，显示市场恐慌情绪"
                elif value < -0.05:
                    return f"波动率偏斜{value:.3f}，显示市场乐观情绪"

        # 默认解释
        return f"{feature}特征值{value:.4f}，对预测有{'正向' if value > 0 else '负向'}影响"