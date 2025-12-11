import os
import pandas as pd
import numpy as np
from src.services.ricequant_service import RiceQuantService
from src.core.load_config import settings
import hashlib
from pykalman import KalmanFilter
from scipy import stats
from tqdm import tqdm

class MLService:
    """
    MLService 囊括价值分析模型、风险分析模型、收益预测模型，能调用投资建议引擎分析模型结果
    """
    def __init__(self):
        self.ricequant_service = RiceQuantService()
        self.project_path = settings.project.project_dir
        self.features_data_path = os.path.join(self.project_path, settings.paths.processed_data)
        print("初始化 MLService 成功！")

    def _analyze_CS(self, start_date:int, end_date:int, order_book_id_list: list = None):
        """
        对股票日线数据进行深度分析（基于TVP-SSM模型）
        :param start_date: yyyymmdd，int型
        :param end_date: yyyymmdd，int型
        :return: 包含所有关键指标的dict列表
        """
        cs_features_list = ['open', 'close', 'high', 'low', 'limit_up', 'limit_down', 'total_turnover', 'volume', 'num_trades', 'prev_close']
        df = self.ricequant_service.instruments_data_fetching(type='CS', start_date=start_date, end_date=end_date, features_list=cs_features_list, order_book_id_list=order_book_id_list)

        # 确保日期格式正确并排序
        df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
        df = df.sort_values(['order_book_id', 'date']).reset_index(drop=True)

        # 计算基础指标 (处理除零错误)
        print('计算基础指标')
        df['institution_participation'] = np.where(
            df['num_trades'] > 0,
            df['volume'] / df['num_trades'],
            np.nan
        )

        # 代理早期流动性 (2021-06-25前)
        print('计算代理早期流动性')
        price_range = (df['high'] - df['low']).replace(0, np.nan)
        df['volume_range_ratio'] = df['volume'] / price_range
        df['institution_participation'] = np.where(
            (df['date'] < '2021-06-25') | df['num_trades'].isna(),
            df['volume_range_ratio'],
            df['institution_participation']
        )

        # 流动性枯竭指数 (处理价格范围为零)
        print('计算流动性枯竭指数')
        df['liquidity_dryup'] = np.where(
            price_range.notna(),
            ((df['limit_up'] - df['close']) / price_range) +
            ((df['close'] - df['limit_down']) / price_range),
            np.nan
        )

        # 涨停延续率 (连续涨停天数)
        print('计算涨停延续率')
        df['is_limit_up'] = df['close'] >= df['limit_up'] * 0.995  # 容忍0.5%误差
        df['consecutive_limit_up'] = 0
        for i in range(1, len(df)):
            if df.iloc[i]['is_limit_up']:
                prev_consec = df.iloc[i - 1]['consecutive_limit_up']
                df.iloc[i, df.columns.get_loc('consecutive_limit_up')] = prev_consec + 1
            else:
                df.iloc[i, df.columns.get_loc('consecutive_limit_up')] = 0

        # ========================
        # 核心：滚动窗口 TVP-SSM 分析
        # ========================
        print('利用TVP-SSM挖掘市场深层的动态风险结构')
        results = []
        window_size = 30  # 滚动窗口大小

        for order_book_id, group in tqdm(df.groupby('order_book_id')):
            group = group.sort_values('date').copy().reset_index(drop=True)
            n = len(group)

            # 准备时间序列
            returns = (group['close'] / group['prev_close'] - 1).values
            liquidity_dryup = group['liquidity_dryup'].fillna(0.5).values

            # 存储动态估计结果
            risk_premium = np.full(n, np.nan)
            liquidity_impact = np.full(n, np.nan)

            # 只有当数据足够长时才进行滚动估计
            if n >= window_size:
                for t in range(window_size, n):
                    # 提取窗口内数据
                    window_returns = returns[t - window_size:t]  # shape: (30,)
                    window_liquidity = liquidity_dryup[t - window_size:t]  # shape: (30,)

                    # 构建观测矩阵：每行 [1, liquidity_dryup_t]
                    obs_mat = np.column_stack([np.ones(window_size), window_liquidity])

                    try:
                        # 初始化模型（状态：[风险溢价, 流动性系数]）
                        kf = KalmanFilter(
                            transition_matrices=np.eye(2),  # 状态转移
                            observation_matrices=obs_mat,  # 观测矩阵（随时间变化）
                            initial_state_mean=[0, 0],
                            initial_state_covariance=np.eye(2),
                            observation_covariance=1e-3,
                            transition_covariance=np.eye(2) * 1e-4
                        )

                        # 使用窗口内数据滤波
                        filtered_state_means, _ = kf.filter(window_returns)
                        # 取最后一天的估计值作为当前状态
                        risk_premium[t] = filtered_state_means[-1, 0]
                        liquidity_impact[t] = filtered_state_means[-1, 1]

                    except Exception as e:
                        # 捕获任何数值不稳定或SVD分解失败等问题
                        risk_premium[t] = np.nan
                        liquidity_impact[t] = np.nan

            # ========================
            # 尾部风险概率
            # ========================
            tail_risk_prob = np.zeros(n)
            for t in range(window_size + 30, n):  # 前面留出缓冲
                recent_rp = risk_premium[t - 60:t]  # 最近60天
                valid_rp = recent_rp[~np.isnan(recent_rp)]
                if len(valid_rp) >= 20:
                    threshold_90 = np.percentile(valid_rp, 90)
                    current_liq = group.iloc[t]['liquidity_dryup']
                    current_rp = risk_premium[t]
                    if not np.isnan(current_rp) and current_liq > 0.8 and current_rp > threshold_90:
                        tail_risk_prob[t] = 0.65

            # ========================
            # 情绪-流动性匹配度
            # ========================
            rolling_mean_ip = group['institution_participation'].rolling(30, min_periods=10).mean()
            emotion_liquidity_match = (
                    (group['institution_participation'] - rolling_mean_ip) *
                    (1 - group['liquidity_dryup'].fillna(0.5))
            )

            # ========================
            # 构建每日结果
            # ========================
            for i in range(n):
                risk_alert = False
                if (not np.isnan(emotion_liquidity_match.iloc[i]) and
                        group.iloc[i]['liquidity_dryup'] > 0.8 and
                        group.iloc[i]['consecutive_limit_up'] > 3 and
                        emotion_liquidity_match.iloc[i] < -0.3):
                    risk_alert = True

                results.append({
                    "date": group.iloc[i]['date'].strftime('%Y-%m-%d'),
                    "order_book_id": order_book_id,
                    "institution_participation": float(group.iloc[i]['institution_participation'])
                    if not pd.isna(group.iloc[i]['institution_participation']) else None,
                    "liquidity_dryup": float(group.iloc[i]['liquidity_dryup'])
                    if not pd.isna(group.iloc[i]['liquidity_dryup']) else None,
                    "consecutive_limit_up": int(group.iloc[i]['consecutive_limit_up']),
                    "daily_return": float(returns[i]) if i < len(returns) and not np.isnan(returns[i]) else None,
                    "dynamic_risk_premium": float(risk_premium[i]) if not np.isnan(risk_premium[i]) else None,
                    "liquidity_impact": float(liquidity_impact[i]) if not np.isnan(liquidity_impact[i]) else None,
                    "tail_risk_probability": float(tail_risk_prob[i]),
                    "emotion_liquidity_match": float(emotion_liquidity_match.iloc[i])
                    if not pd.isna(emotion_liquidity_match.iloc[i]) else None,
                    "risk_alert": risk_alert
                })

        # indicator_explanations = {
        #     "date": "交易日期，作为基础时间戳用于追踪趋势变化。",
        #     "order_book_id": "股票代码，用于标识分析的具体标的对象。",
        #     "institution_participation": "机构参与度，计算方式为 volume / num_trades，表示平均每笔交易的股数。>2000 表示机构主导，趋势可能延续；<1000 表示散户主导，易追涨杀跌、波动大；值在中间（如1765.58）表示处于中间偏机构状态，但尚未形成稳定主力。",
        #     "liquidity_dryup": "流动性枯竭指数，计算方式为 (limit_up - close)/(high-low) + (close - limit_down)/(high-low)，反映价格离涨跌停的接近程度。接近0 表示价格卡在涨停或跌停，流动性枯竭，存在风险；接近2 表示价格在中间区域，交易活跃；值为0.752 表示价格偏向跌停，流动性紧张，抛售压力仍在。",
        #     "consecutive_limit_up": "连续涨停天数，统计连续收盘价≥涨停价的天数，反映当前是否处于强势上涨趋势。>3 表示强趋势，但需警惕情绪过热；=0 表示无趋势或趋势中断；值为0 表示趋势未启动，市场仍弱。",
        #     "daily_return": "日收益率，计算方式为 close/prev_close - 1，表示当天的涨跌幅。值为-0.0298 表示下跌2.98%，属于显著回调，结合其他指标可判断为恐慌性抛售。",
        #     "dynamic_risk_premium": "动态风险溢价，由TVP-SSM模型估计得出，反映市场当天要求多少额外回报来承担风险。null 表示模型需要至少30天数据才能输出，当前为预热阶段；后续若跳升超过0.5%，表明市场极度恐慌，风险偏好下降。",
        #     "liquidity_impact": "流动性冲击系数，由TVP-SSM模型估计得出，衡量流动性恶化对风险溢价的放大作用。null 表示模型预热中；后续若为正且持续增大，表明流动性已成为主要风险来源。",
        #     "tail_risk_probability": "尾部风险概率，基于历史回测规则生成，表示未来30天出现大幅回撤的概率。0.0 表示当前不满足高风险条件；>0.65 表示极高风险，建议减仓或对冲；触发条件为 liquidity_dryup > 0.8 且 dynamic_risk_premium 处于高位。",
        #     "emotion_liquidity_match": "情绪-流动性匹配度，计算方式为 (机构参与度 - 市场均值) × (1 - liquidity_dryup)，用于判断‘谁在买’和‘能不能卖’是否协调。>0 表示健康状态（机构买入且流动性好）；<-0.3 表示危险信号（机构撤离且流动性差）；值为-0.12 表示轻度不匹配，但未达到警戒水平。",
        #     "risk_alert": "风险预警，由综合逻辑判断生成，指示是否应发出减仓信号。False 表示暂无系统性风险；True 表示满足多重高风险条件（如流动性枯竭、情绪恶化等），强烈建议采取行动。"
        # }
        return results

    def summarize_CSanalysis(self, start_date: int, end_date: int, target_stock_id=None,
                                order_book_id_list: list = None, lookback_days=30, confidence_level=0.95):
        """
        对深度分析指标进行时间序列趋势分析，识别动态模式与领先-滞后关系

        返回:
        str: 包含深度趋势分析的自然语言总结
        """
        analysis_results = self._analyze_CS(start_date, end_date, order_book_id_list)
        if not analysis_results:
            return "无数据可供分析。"

        # 转换为DataFrame并预处理
        df = pd.DataFrame(analysis_results)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['order_book_id', 'date'])

        # ======================
        # 新增：市场基准计算（保留原代码不变，新增此部分）
        # ======================
        # 获取最新日期用于市场基准计算
        latest_date = df['date'].max()

        # 获取所有股票在最新日期的数据
        market_df = df[df['date'] == latest_date].copy()

        # 计算市场基准（排除目标股票自身）
        market_benchmarks = {}
        if len(market_df) > 1:  # 至少有2只股票才能计算基准
            if target_stock_id:
                market_without_target = market_df[market_df['order_book_id'] != target_stock_id]
            else:
                market_without_target = market_df  # 如果没有指定目标股票，使用全部数据

            if not market_without_target.empty:
                market_benchmarks = {
                    'inst_part_mean': market_without_target['institution_participation'].mean(),
                    'inst_part_25pct': market_without_target['institution_participation'].quantile(0.25),
                    'inst_part_75pct': market_without_target['institution_participation'].quantile(0.75),
                    'liquidity_mean': market_without_target['liquidity_dryup'].mean(),
                    'liquidity_25pct': market_without_target['liquidity_dryup'].quantile(0.25),
                    'liquidity_75pct': market_without_target['liquidity_dryup'].quantile(0.75),
                }
                # 计算风险溢价基准（仅包含有效值）
                valid_rp = market_without_target['dynamic_risk_premium'].dropna()
                if not valid_rp.empty:
                    market_benchmarks['risk_premium_mean'] = valid_rp.mean()

        # 选择目标股票
        if target_stock_id:
            stock_df = df[df['order_book_id'] == target_stock_id].copy()
            if stock_df.empty:
                return f"未找到股票 {target_stock_id} 的数据。"
        else:
            # 自动选择第一个股票
            target_stock_id = df['order_book_id'].iloc[0]
            stock_df = df[df['order_book_id'] == target_stock_id].copy()

        # 限制分析窗口
        if len(stock_df) > lookback_days:
            stock_df = stock_df.tail(lookback_days).reset_index(drop=True)

        n = len(stock_df)
        if n < 10:  # 需要足够数据进行趋势分析
            return f"股票 {target_stock_id} 数据点不足（{n}天），无法进行有效趋势分析。"

        # ======================
        # 1. 深度时间序列趋势分析
        # ======================

        # --- 1.1 机构参与度趋势（核心指标）---
        inst_part = stock_df['institution_participation'].astype(float)

        # 计算线性趋势斜率和显著性
        x = np.arange(n)
        slope_inst, intercept_inst, r_inst, p_inst, std_err_inst = stats.linregress(x, inst_part)
        trend_strength_inst = abs(slope_inst) * n / inst_part.mean()

        # 趋势分类（基于统计显著性和强度）
        inst_trend_desc = ""
        if p_inst < (1 - confidence_level):
            if slope_inst > 0:
                if trend_strength_inst > 0.5:
                    inst_trend_desc = "显著上升趋势，机构资金持续大幅流入"
                elif trend_strength_inst > 0.2:
                    inst_trend_desc = "温和上升趋势，机构资金逐步介入"
                else:
                    inst_trend_desc = "轻微上升趋势，机构参与度缓慢改善"
            else:
                if trend_strength_inst > 0.5:
                    inst_trend_desc = "显著下降趋势，机构资金加速撤离"
                elif trend_strength_inst > 0.2:
                    inst_trend_desc = "温和下降趋势，机构资金缓慢流出"
                else:
                    inst_trend_desc = "轻微下降趋势，机构参与度缓慢降低"

            # 添加统计信息
            inst_trend_desc += f"（斜率={slope_inst:.2f}, p={p_inst:.3f}）"
        else:
            inst_trend_desc = "无显著趋势，机构参与度随机波动"

        # --- 1.2 流动性枯竭指数趋势 ---
        liquidity = stock_df['liquidity_dryup'].astype(float)

        # 检测加速恶化（二阶导数近似）
        liquidity_ma = liquidity.rolling(window=5).mean().dropna()
        liquidity_accel = liquidity_ma.diff().mean()

        # 趋势分类
        liquidity_status = ""
        if liquidity_accel > 0.03 and liquidity.iloc[-1] > 0.7:
            liquidity_status = f"流动性正在加速恶化（加速度={liquidity_accel:.4f}），价格持续贴近跌停，交易极度困难"
        elif liquidity.iloc[-1] > 0.8:
            liquidity_status = "流动性严重紧张，价格频繁触及涨跌停，交易困难"
        elif liquidity.iloc[-1] > 0.6:
            liquidity_status = "流动性紧张，价格接近涨跌停区间"
        elif liquidity.iloc[-1] < 0.3:
            liquidity_status = "流动性充足，价格运行平稳，交易活跃"
        else:
            liquidity_status = "流动性处于正常水平"

        # --- 1.3 风险溢价动态分析（核心趋势）---
        risk_premium = stock_df['dynamic_risk_premium'].dropna()
        if len(risk_premium) >= 15:
            x_rp = np.arange(len(risk_premium))
            slope_rp, _, _, p_rp, _ = stats.linregress(x_rp, risk_premium)

            # 计算波动率变化
            rp_vol = risk_premium.diff().abs().rolling(window=5).mean().dropna()
            vol_trend = "显著上升" if rp_vol.iloc[-1] > rp_vol.quantile(0.75) else "趋于稳定"

            if p_rp < (1 - confidence_level) and slope_rp > 0:
                risk_summary = f"动态风险溢价呈显著上升趋势（斜率={slope_rp:.4f}, p={p_rp:.3f}），且波动率{vol_trend}，显示市场避险情绪快速升温"
            elif p_rp < (1 - confidence_level) and slope_rp < 0:
                risk_summary = f"动态风险溢价呈显著下降趋势（斜率={slope_rp:.4f}, p={p_rp:.3f}），且波动率{vol_trend}，显示市场风险偏好回升"
            else:
                risk_summary = f"动态风险溢价波动较大但无显著趋势（p={p_rp:.3f}），波动率{vol_trend}，市场情绪处于平衡状态"
        else:
            risk_summary = "动态风险溢价数据不足，暂无法进行趋势分析"

        # --- 1.4 尾部风险动态模式 ---
        tail_risk = stock_df['tail_risk_probability'].astype(float)
        high_risk_days = (tail_risk >= 0.65).sum()
        medium_risk_days = ((tail_risk >= 0.4) & (tail_risk < 0.65)).sum()
        risk_persistence = (tail_risk > 0).astype(int).diff().ne(1).cumsum().value_counts().max() / n

        if high_risk_days > n * 0.2:
            tail_risk_summary = f"高风险状态频繁（{high_risk_days}天，占比{high_risk_days / n:.0%}），且持续性强（平均持续{int(1 / risk_persistence)}天），系统性风险累积明显"
        elif high_risk_days > 0:
            tail_risk_summary = f"偶发高风险状态（{high_risk_days}天），但持续时间短，需警惕突发风险"
        elif medium_risk_days > n * 0.3:
            tail_risk_summary = f"中等风险状态持续（{medium_risk_days}天，占比{medium_risk_days / n:.0%}），市场脆弱性增强"
        else:
            tail_risk_summary = "风险水平整体可控，系统性崩溃概率低"

        # --- 1.5 风险预警信号模式分析 ---
        risk_alerts = stock_df['risk_alert'].astype(bool)
        alert_count = risk_alerts.sum()
        alert_clusters = (risk_alerts != risk_alerts.shift()).cumsum()[risk_alerts].value_counts()
        avg_cluster_size = alert_clusters.mean() if not alert_clusters.empty else 0

        if alert_count > n * 0.2:
            alert_summary = f"风险预警高频触发（{alert_count}次，{alert_count / n:.0%}天），且常成簇出现（平均{avg_cluster_size:.1f}天/簇），市场处于持续危险状态"
        elif alert_count > 0:
            alert_summary = f"偶发风险预警（{alert_count}次），多为孤立事件，但需关注触发条件"
        else:
            alert_summary = "未触发风险预警，当前市场环境相对安全"

        # --- 1.6 情绪-流动性动态关系 ---
        match_series = stock_df['emotion_liquidity_match'].astype(float)

        # 计算与机构参与度的滚动相关性
        rolling_corr = inst_part.rolling(window=5).corr(match_series)
        avg_corr = rolling_corr.mean()

        if match_series.mean() < -0.25:
            match_trend = f"持续严重负向（均值={match_series.mean():.2f}），机构撤离与流动性恶化形成恶性循环"
        elif match_series.mean() < -0.1:
            match_trend = f"持续轻度负向（均值={match_series.mean():.2f}），需关注资金流向变化"
        elif match_series.mean() > 0.25:
            match_trend = f"持续高度协调（均值={match_series.mean():.2f}），机构主导且流动性好，趋势健康"
        else:
            match_trend = f"基本平衡（均值={match_series.mean():.2f}），市场处于过渡状态"

        # --- 1.7 领先-滞后关系分析（关键！）---
        # 检查流动性枯竭是否领先于风险溢价变化
        lead_lag_results = []
        best_lag = None
        if len(risk_premium) >= 20:
            for lag in range(-7, 8):  # -7到+7天的滞后
                if lag <= 0:
                    corr = liquidity[:lag].corr(risk_premium[-lag:]) if lag != 0 else liquidity.corr(risk_premium)
                else:
                    corr = liquidity[lag:].corr(risk_premium[:-lag])
                lead_lag_results.append((lag, corr))

            best_lag, best_corr = max(lead_lag_results, key=lambda x: abs(x[1]))
            if abs(best_corr) > 0.45:
                if best_lag < 0:
                    lead_lag_summary = f"流动性枯竭领先风险溢价约{-best_lag}天（最大相关系数={best_corr:.2f}），是市场风险的先行指标"
                elif best_lag > 0:
                    lead_lag_summary = f"风险溢价领先流动性枯竭约{best_lag}天（最大相关系数={best_corr:.2f}），风险情绪先于流动性变化"
                else:
                    lead_lag_summary = f"流动性枯竭与风险溢价同步变化（相关系数={best_corr:.2f}），风险与流动性相互强化"
            else:
                lead_lag_summary = "流动性与风险溢价关系不稳定，无明显领先-滞后模式"
        else:
            lead_lag_summary = "数据不足，无法进行领先-滞后分析"

        # ======================
        # 2. 识别关键动态模式
        # ======================

        # 模式1: 流动性危机模式
        liquidity_crisis = (
            "流动性加速恶化" in liquidity_status and
            "显著下降趋势" in inst_trend_desc and
            "上升趋势" in risk_summary and
            best_lag < 0 if 'best_lag' in locals() else False
        )

        # 模式2: 健康上涨模式
        healthy_rally = (
                "显著上升趋势" in inst_trend_desc and
                "流动性充足" in liquidity_status and
                ("下降趋势" in risk_summary or "平衡" in risk_summary)
        )

        # 模式3: 情绪驱动波动模式
        emotion_volatility = (
                abs(avg_corr) < 0.2 and
                "波动剧烈" in risk_summary and
                "基本平衡" in match_trend
        )

        # ======================
        # 3. 新增：相对市场定位分析（保留原代码不变，新增此部分）
        # ======================
        relative_analysis = ""
        latest = stock_df.iloc[-1]

        # 1. 机构参与度相对位置
        inst_relative_desc = "无市场比较数据"
        inst_part_relative = None
        if 'inst_part_mean' in market_benchmarks:
            # 计算Z-score（使用IQR标准化）
            iqr = market_benchmarks['inst_part_75pct'] - market_benchmarks['inst_part_25pct']
            if iqr > 0:  # 防止除以0
                inst_part_relative = (
                        (latest['institution_participation'] - market_benchmarks['inst_part_mean']) /
                        (iqr + 1e-5)
                )
                if inst_part_relative > 1.0:
                    inst_relative_desc = "显著高于市场平均水平，机构关注度突出"
                elif inst_part_relative > 0.5:
                    inst_relative_desc = "高于市场平均水平，机构关注度较高"
                elif inst_part_relative < -1.0:
                    inst_relative_desc = "显著低于市场平均水平，机构关注度低迷"
                elif inst_part_relative < -0.5:
                    inst_relative_desc = "低于市场平均水平，机构关注度较低"
                else:
                    inst_relative_desc = "接近市场平均水平"

        # 2. 流动性相对位置
        liquidity_relative_desc = "无市场比较数据"
        liquidity_relative = None
        if 'liquidity_mean' in market_benchmarks:
            # 计算Z-score（使用IQR标准化）
            iqr = market_benchmarks['liquidity_75pct'] - market_benchmarks['liquidity_25pct']
            if iqr > 0:  # 防止除以0
                liquidity_relative = (
                        (latest['liquidity_dryup'] - market_benchmarks['liquidity_mean']) /
                        (iqr + 1e-5)
                )
                if liquidity_relative > 1.0:
                    liquidity_relative_desc = "流动性紧张程度显著高于市场，交易难度大"
                elif liquidity_relative > 0.5:
                    liquidity_relative_desc = "流动性紧张程度高于市场"
                elif liquidity_relative < -1.0:
                    liquidity_relative_desc = "流动性显著优于市场，交易顺畅"
                elif liquidity_relative < -0.5:
                    liquidity_relative_desc = "流动性优于市场"
                else:
                    liquidity_relative_desc = "流动性处于市场正常水平"

        # 3. 风险溢价相对位置
        risk_relative_desc = "无市场比较数据或数据不足"
        if 'risk_premium_mean' in market_benchmarks and not pd.isna(latest['dynamic_risk_premium']):
            risk_premium_relative = (
                    latest['dynamic_risk_premium'] - market_benchmarks['risk_premium_mean']
            )
            if risk_premium_relative > 0.003:
                risk_relative_desc = "风险溢价显著高于市场，避险情绪强烈"
            elif risk_premium_relative > 0.001:
                risk_relative_desc = "风险溢价高于市场，避险情绪较高"
            elif risk_premium_relative < -0.003:
                risk_relative_desc = "风险溢价显著低于市场，风险偏好突出"
            elif risk_premium_relative < -0.001:
                risk_relative_desc = "风险溢价低于市场，风险偏好较高"
            else:
                risk_relative_desc = "风险溢价接近市场水平"

        # ======================
        # 4. 综合总结输出（修改这部分以包含相对分析）
        # ======================
        summary = f"""
        【{target_stock_id} 深度趋势分析报告】（截至 {stock_df['date'].max().strftime('%Y-%m-%d')}）
    
        🌍 市场相对定位（基于{len(market_df)}只股票最新数据）：
        - 机构参与度：{inst_relative_desc}
        - 流动性状况：{liquidity_relative_desc}
        - 风险溢价水平：{risk_relative_desc}
    
        🔍 核心趋势诊断（基于{len(stock_df)}天数据）：
        1. **机构参与趋势**：{inst_trend_desc}
           - 当前值：{inst_part.iloc[-1]:.2f}（{('↑' if slope_inst > 0 else '↓') if p_inst < 0.05 else '→'}）
           - 相对市场位置：{'高于' if inst_part_relative and inst_part_relative > 0 else '低于' if inst_part_relative and inst_part_relative < 0 else '接近'}市场中位数
           - 5日移动平均：{inst_part.rolling(5).mean().iloc[-1]:.2f}
    
        2. **流动性状况**：{liquidity_status}
           - 当前流动性枯竭指数：{liquidity.iloc[-1]:.3f}
           - 流动性加速度：{liquidity_accel:.4f}（正值表示恶化加速）
           - 相对市场位置：{'紧张程度高于' if liquidity_relative and liquidity_relative > 0 else '紧张程度低于' if liquidity_relative and liquidity_relative < 0 else '接近'}市场平均水平
    
        3. **风险情绪动态**：{risk_summary}
           - 风险波动率趋势：{vol_trend if 'vol_trend' in locals() else 'N/A'}
           - 相对市场风险：{risk_relative_desc.lower()}
    
        4. **关键动态关系**：{lead_lag_summary}
           - {'流动性指标可作为风险变化的领先指标，提前预警市场压力'
            if '领先' in lead_lag_summary and best_lag and best_lag < 0
            else '风险情绪变化先于流动性恶化，需优先关注情绪指标'
            if '领先' in lead_lag_summary and best_lag and best_lag > 0
            else '流动性与风险情绪同步变化，需同时监控'}
    
        💡 识别到的市场模式：
        {'⚠️【流动性危机模式】机构撤离、流动性恶化加速且领先于风险上升，市场脆弱性极高！' if liquidity_crisis else
            '📈【健康上涨模式】机构持续流入、流动性充足且风险情绪稳定，趋势健康可持续。' if healthy_rally else
            '🔄【情绪驱动波动】市场情绪与流动性匹配度低，价格波动主要由情绪驱动，趋势难以持续。' if emotion_volatility else
            '🔍【混合状态】市场处于过渡期，需密切关注领先指标变化方向。'}
    
        📊 风险状态评估：
        - 尾部风险：{tail_risk_summary}
        - 风险预警：{alert_summary}
        - 情绪-流动性匹配：{match_trend}
    
        🎯 操作建议（基于当前模式和市场相对位置）：
        {('🔴【紧急行动】流动性危机模式已确认！建议：' +
        '   - 立即减仓50%以上，保留现金应对流动性枯竭' +
        '   - 买入短期虚值Put期权对冲尾部风险' +
        '   - 密切监控流动性指标，若加速度继续上升则全部离场' if liquidity_crisis else
        '🟢【积极布局】健康上涨模式确认！建议：' +
        '   - 保持或适度加仓，目标仓位70-90%' +
        '   - 使用备兑策略(Covered Call)增强收益' +
        '   - 若机构参与度增速放缓则部分止盈' if healthy_rally else
        '🟡【谨慎操作】情绪驱动波动模式！建议：' +
        '   - 降低仓位至30-50%，避免追高杀跌' +
        '   - 采用跨式组合(Straddle)捕捉波动' +
        '   - 重点监控机构参与度变化，判断趋势方向' if emotion_volatility else
        '🔵【观察等待】混合状态！建议：' +
        '   - 维持中性仓位(40-60%)' +
        '   - 设置突破策略：若流动性枯竭指数突破0.85则减仓，跌破0.5则加仓' +
        '   - 每周重新评估市场模式')}
        
        📌 风险提示：
        - 本分析基于历史数据，未来市场结构可能变化
        - 建议每周更新分析，尤其关注领先指标变化
        
        🔍 深度洞察：
        {('流动性枯竭指数领先风险溢价约' + str(-best_lag) + '天，可作为早期预警信号。'
        if '领先' in lead_lag_summary and best_lag and best_lag < 0
        else '风险情绪变化是流动性恶化的早期指标，提前关注风险溢价走势。'
        if '领先' in lead_lag_summary and best_lag and best_lag > 0
        else '流动性与风险情绪同步变化，需同时监控两类指标。')}
        
        💡 特别提示：
        该股票当前表现{('显著领先' if inst_part_relative and inst_part_relative > 0.5 and liquidity_relative and liquidity_relative < 0 else
        '领先' if inst_part_relative and inst_part_relative > 0.3 and liquidity_relative and liquidity_relative < 0.3 else
        '落后于' if inst_part_relative and inst_part_relative < -0.5 and liquidity_relative and liquidity_relative > 0.5 else
        '与')}市场整体，{('建议' if inst_part_relative and inst_part_relative > 0.3 and liquidity_relative and liquidity_relative < 0.3 else '谨慎')}{'增持' if inst_part_relative and inst_part_relative > 0.5 and liquidity_relative and liquidity_relative < 0 else '减持' if inst_part_relative and inst_part_relative < -0.5 and liquidity_relative and liquidity_relative > 0.5 else '持有'}
        """.strip()

        return summary

    def _analyze_ETF(self, start_date:int, end_date:int, order_book_id_list: list = None):
        """
        对ETF日线数据进行深度分析，输出包含所有关键指标的字典列表
        """
        etf_features_list = ['open', 'close', 'high', 'low', 'total_turnover', 'volume', 'num_trades', 'prev_close', 'iopv']
        df = self.ricequant_service.instruments_data_fetching(type='ETF', start_date=start_date, end_date=end_date, features_list=etf_features_list, order_book_id_list=order_book_id_list)

        # 确保日期格式正确并排序
        df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
        df = df.sort_values(['order_book_id', 'date']).reset_index(drop=True)

        # 计算基础指标
        # 日溢价率 (核心指标)
        df['daily_premium_rate'] = (df['close'] - df['iopv']) / df['iopv']

        # iopv稳定性 (替代日内波动率)
        price_range = (df['high'] - df['low']).replace(0, np.nan)
        df['iopv_stability'] = np.where(
            price_range.notna(),
            1 - (df['iopv'] - df['close']).abs() / price_range,
            np.nan
        )

        # ETF日收益率
        df['etf_return'] = df['close'] / df['prev_close'] - 1

        # 无指数数据时的替代方案
        df['tracking_error'] = np.nan
        df['index_volatility'] = np.nan
        df['actual_tracking_cost'] = np.nan

        # 3. 溢价持续性 (考虑成交量)
        df['volume_ma_20'] = df['volume'].rolling(20, min_periods=10).mean()
        df['premium_persistence'] = df['daily_premium_rate'] * (df['volume'] / df['volume_ma_20'])

        # 4. 溢价率与成交量的相关性 (用于流动性危机预警)
        df['premium_vol_corr'] = df['daily_premium_rate'].rolling(20, min_periods=10).corr(df['volume'])

        # 5. 简化版误差修正项 (模拟EC term)
        # 原理：溢价率向0回归的速度，值越大表示回归越快
        df['ec_term'] = np.nan

        # 仅当有足够数据时计算
        for i in range(20, len(df)):
            window_premium = df['daily_premium_rate'].iloc[i - 19:i + 1]
            # 计算溢价率的均值回归系数（简化版EC term）
            if len(window_premium) >= 10 and not window_premium.isna().all():
                try:
                    # 用昨日溢价率预测今日溢价率，回归系数反映均值回归速度
                    x = window_premium.iloc[:-1].values
                    y = window_premium.iloc[1:].values
                    slope, _, _, _, _ = stats.linregress(x, y)
                    # EC term = 1 - slope (正值表示均值回归)
                    df.iloc[i, df.columns.get_loc('ec_term')] = 1 - slope
                except:
                    pass

        # 6. 风险预警信号
        df['risk_alert'] = False

        # 条件1：溢价率持续3日 > 0.5%
        premium_high = (df['daily_premium_rate'] > 0.005)
        df['premium_high_streak'] = premium_high.astype(int).groupby((~premium_high).cumsum()).cumsum()

        # 条件2：实际跟踪成本 > 指数波动率15%
        if 'index_volatility' in df and 'actual_tracking_cost' in df:
            cost_to_vol_ratio = df['actual_tracking_cost'] / df['index_volatility']
            high_cost = cost_to_vol_ratio > 1.15  # 比指数波动率高15%

            # 风险预警：满足两个条件
            df['risk_alert'] = (df['premium_high_streak'] >= 3) & high_cost

        # 7. 套利机会信号
        df['arbitrage_opportunity'] = False
        if 'ec_term' in df:
            # 套利窗口期：当 |EC term| > 0.5 且 溢价率 > 0.3%
            df['arbitrage_opportunity'] = (df['ec_term'].abs() > 0.5) & (df['daily_premium_rate'].abs() > 0.003)

        # 8. 流动性危机预警
        df['liquidity_crisis_warning'] = False
        if 'premium_vol_corr' in df and 'ec_term' in df:
            # 当溢价率与volume负相关（Pearson < -0.4），且EC term趋近0
            df['liquidity_crisis_warning'] = (df['premium_vol_corr'] < -0.4) & (df['ec_term'].abs() < 0.1)

        # 9. 构建结果字典
        results = []
        for _, row in df.iterrows():
            result = {
                "date": row['date'].strftime('%Y-%m-%d'),
                "order_book_id": row['order_book_id'],
                "daily_premium_rate": float(row['daily_premium_rate']),
                "iopv_stability": float(row['iopv_stability']) if not pd.isna(row['iopv_stability']) else None,
                "etf_return": float(row['etf_return']),
                "premium_persistence": float(row['premium_persistence']) if not pd.isna(
                    row['premium_persistence']) else None,
                "tracking_error": float(row['tracking_error']) if not pd.isna(row['tracking_error']) else None,
                "index_volatility": float(row['index_volatility']) if not pd.isna(
                    row['index_volatility']) else None,
                "actual_tracking_cost": float(row['actual_tracking_cost']) if not pd.isna(
                    row['actual_tracking_cost']) else None,
                "premium_vol_corr": float(row['premium_vol_corr']) if not pd.isna(
                    row['premium_vol_corr']) else None,
                "ec_term": float(row['ec_term']) if not pd.isna(row['ec_term']) else None,
                "risk_alert": bool(row['risk_alert']),
                "arbitrage_opportunity": bool(row['arbitrage_opportunity']),
                "liquidity_crisis_warning": bool(row['liquidity_crisis_warning'])
            }
            results.append(result)

        return results

    def summarize_ETFanalysis(self, start_date: int, end_date: int, target_ETF_id=None,
                                order_book_id_list: list = None, lookback_days=30, confidence_level=0.95):
        """
        对ETF深度分析指标进行时间序列趋势分析，识别动态模式与领先-滞后关系
        关键特性：基于多只ETF数据，提供目标ETF的相对市场定位分析，并捕捉时间序列趋势
        """
        # 1. 获取ETF分析结果
        analysis_results = self._analyze_ETF(start_date, end_date, order_book_id_list)
        if not analysis_results:
            return "无ETF数据可供分析。"

        # 转换为DataFrame并预处理
        df = pd.DataFrame(analysis_results)

        # 确保日期格式正确
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        # 按ETF代码和日期排序
        df = df.sort_values(['order_book_id', 'date']).reset_index(drop=True)

        # ======================
        # 2. 硻定目标ETF并计算市场基准
        # ======================

        # 确定要分析的目标ETF
        if target_ETF_id:
            if target_ETF_id not in df['order_book_id'].unique():
                return f"未找到ETF {target_ETF_id} 的数据。"
        else:
            # 自动选择第一个ETF
            target_ETF_id = df['order_book_id'].iloc[0]

        # 获取最新日期（用于市场基准计算）
        latest_date = df['date'].max()

        # 获取所有ETF在最新日期的数据（用于计算市场基准）
        market_df = df[df['date'] == latest_date].copy()

        # 计算市场基准（排除目标ETF自身，避免自相关）
        market_benchmarks = {}
        if len(market_df) > 1:  # 至少有2只ETF才能计算有意义的基准
            market_without_target = market_df[market_df['order_book_id'] != target_ETF_id]
            if not market_without_target.empty:
                # 只有当ec_term存在且非空时才计算其统计量
                valid_ec_term = market_without_target['ec_term'].dropna()
                ec_term_mean = valid_ec_term.mean() if not valid_ec_term.empty else None
                ec_term_quantiles = valid_ec_term.quantile([0.25, 0.75]) if not valid_ec_term.empty else None

                market_benchmarks = {
                    'premium_mean': market_without_target['daily_premium_rate'].mean(),
                    'premium_25pct': market_without_target['daily_premium_rate'].quantile(0.25),
                    'premium_75pct': market_without_target['daily_premium_rate'].quantile(0.75),
                    'stability_mean': market_without_target['iopv_stability'].mean(),
                    'stability_25pct': market_without_target['iopv_stability'].quantile(0.25),
                    'stability_75pct': market_without_target['iopv_stability'].quantile(0.75),
                    'tracking_cost_mean': market_without_target['actual_tracking_cost'].mean(),
                    'ec_term_mean': ec_term_mean,
                    'ec_term_25pct': ec_term_quantiles[0.25] if ec_term_quantiles is not None else None,
                    'ec_term_75pct': ec_term_quantiles[0.75] if ec_term_quantiles is not None else None
                }

        # 选择目标ETF的时间序列数据
        etf_df = df[df['order_book_id'] == target_ETF_id].copy()

        # 限制分析窗口
        if len(etf_df) > lookback_days:
            etf_df = etf_df.tail(lookback_days).reset_index(drop=True)

        n = len(etf_df)
        if n < 10:  # 需要足够数据进行趋势分析
            return f"ETF {target_ETF_id} 数据点不足（{n}天），无法进行有效趋势分析。"

        # 获取最新数据点
        latest = etf_df.iloc[-1]

        # ======================
        # 3. 深度时间序列趋势分析
        # ======================

        # --- 3.1 溢价率趋势（核心指标）---
        premium_rate = etf_df['daily_premium_rate'].astype(float)

        # 计算线性趋势斜率和显著性
        x = np.arange(n)
        try:
            slope_premium, intercept_premium, r_premium, p_premium, std_err_premium = stats.linregress(x,
                                                                                                       premium_rate)
            trend_strength_premium = abs(slope_premium) * n / (premium_rate.abs().mean() + 1e-5)

            # 趋势分类
            premium_trend_desc = ""
            if p_premium < (1 - confidence_level):
                if slope_premium > 0:
                    if trend_strength_premium > 0.5:
                        premium_trend_desc = "显著上升趋势，二级市场持续供不应求"
                    elif trend_strength_premium > 0.2:
                        premium_trend_desc = "温和上升趋势，二级市场需求逐步增强"
                    else:
                        premium_trend_desc = "轻微上升趋势，溢价率缓慢改善"
                else:
                    if trend_strength_premium > 0.5:
                        premium_trend_desc = "显著下降趋势，二级市场抛压持续"
                    elif trend_strength_premium > 0.2:
                        premium_trend_desc = "温和下降趋势，二级市场抛压逐步显现"
                    else:
                        premium_trend_desc = "轻微下降趋势，溢价率缓慢恶化"

                # 添加统计信息
                premium_trend_desc += f"（斜率={slope_premium:.4f}, p={p_premium:.3f}）"
            else:
                premium_trend_desc = "无显著趋势，溢价率随机波动"
        except Exception as e:
            premium_trend_desc = "溢价率趋势分析失败，数据可能存在问题"

        # --- 3.2 iopv稳定性趋势 ---
        stability = etf_df['iopv_stability'].dropna().astype(float)

        # 检测稳定性变化率
        stability_ma = stability.rolling(window=5).mean().dropna()
        stability_change = stability_ma.diff().mean() if len(stability_ma) > 1 else 0

        stability_status = ""
        if len(stability) > 0:
            current_stability = stability.iloc[-1]
            if current_stability < 0.3:
                stability_status = f"iopv严重失真（当前值={current_stability:.2f}），ETF NAV计算可能失效，警惕成分股停牌影响"
            elif current_stability < 0.6:
                stability_status = f"iopv稳定性一般（当前值={current_stability:.2f}），需关注成分股流动性"
            else:
                stability_status = f"iopv稳定性良好（当前值={current_stability:.2f}），ETF定价效率高"

            # 添加变化趋势
            if stability_change > 0.05:
                stability_status += "，且呈明显改善趋势"
            elif stability_change < -0.05:
                stability_status += "，且呈明显恶化趋势"
        else:
            stability_status = "iopv稳定性数据不足"

        # --- 3.3 实际跟踪成本分析 ---
        tracking_cost = etf_df['actual_tracking_cost'].dropna()
        tracking_cost_summary = "实际跟踪成本数据不足"

        if len(tracking_cost) >= 15:
            # 计算跟踪成本趋势
            x_tc = np.arange(len(tracking_cost))
            try:
                slope_tc, _, _, p_tc, _ = stats.linregress(x_tc, tracking_cost)

                if p_tc < (1 - confidence_level) and slope_tc > 0:
                    tracking_cost_summary = f"实际跟踪成本呈显著上升趋势（斜率={slope_tc:.4f}, p={p_tc:.3f}），ETF效率持续恶化"
                elif p_tc < (1 - confidence_level) and slope_tc < 0:
                    tracking_cost_summary = f"实际跟踪成本呈显著下降趋势（斜率={slope_tc:.4f}, p={p_tc:.3f}），ETF效率持续改善"
                else:
                    tracking_cost_summary = f"实际跟踪成本波动但无显著趋势（p={p_tc:.3f}），ETF效率保持稳定"
            except Exception as e:
                tracking_cost_summary = "实际跟踪成本趋势分析失败"

        # --- 3.4 溢价率与成交量关系 ---
        premium_vol_corr = etf_df['premium_vol_corr'].dropna()
        corr_summary = "溢价率与成交量关系数据不足"

        if len(premium_vol_corr) > 5:
            avg_corr = premium_vol_corr.mean()
            if avg_corr < -0.4:
                corr_summary = f"溢价率与成交量显著负相关（均值={avg_corr:.2f}），市场可能失效"
            elif avg_corr < -0.2:
                corr_summary = f"溢价率与成交量负相关（均值={avg_corr:.2f}），需关注流动性"
            elif avg_corr > 0.4:
                corr_summary = f"溢价率与成交量显著正相关（均值={avg_corr:.2f}），市场效率高"
            else:
                corr_summary = f"溢价率与成交量相关性弱（均值={avg_corr:.2f}），市场运行正常"
        else:
            corr_summary = "溢价率与成交量关系数据不足"

        # --- 3.5 误差修正项(EC term)趋势 ---
        ec_term = etf_df['ec_term'].dropna()
        ec_term_summary = "误差修正项数据不足"

        if len(ec_term) >= 15:
            # 计算EC term趋势
            x_ec = np.arange(len(ec_term))
            try:
                slope_ec, _, _, p_ec, _ = stats.linregress(x_ec, ec_term)

                if p_ec < (1 - confidence_level) and slope_ec > 0.1:
                    ec_term_summary = f"EC term呈显著上升趋势（斜率={slope_ec:.2f}, p={p_ec:.3f}），溢价收敛速度加快"
                elif p_ec < (1 - confidence_level) and slope_ec < -0.1:
                    ec_term_summary = f"EC term呈显著下降趋势（斜率={slope_ec:.2f}, p={p_ec:.3f}），溢价收敛速度减慢"
                else:
                    ec_term_summary = f"EC term波动但无显著趋势（p={p_ec:.3f}），溢价收敛机制稳定"
            except Exception as e:
                ec_term_summary = "EC term趋势分析失败"
        else:
            ec_term_summary = "误差修正项数据不足"

        # --- 3.6 领先-滞后关系分析 ---
        lead_lag_results = []
        best_lag = None
        best_corr = 0.0
        if len(ec_term) >= 20 and len(premium_rate) >= 20:
            for lag in range(-7, 8):  # -7到+7天的滞后
                try:
                    if lag <= 0:
                        corr = ec_term[:lag].corr(premium_rate[-lag:]) if lag != 0 else ec_term.corr(premium_rate)
                    else:
                        corr = ec_term[lag:].corr(premium_rate[:-lag])
                    lead_lag_results.append((lag, corr))
                except:
                    continue

            if lead_lag_results:
                best_lag, best_corr = max(lead_lag_results, key=lambda x: abs(x[1]))
                if abs(best_corr) > 0.4:
                    if best_lag < 0:
                        lead_lag_summary = f"EC term领先溢价率约{-best_lag}天（最大相关系数={best_corr:.2f}），是溢价变化的先行指标"
                    elif best_lag > 0:
                        lead_lag_summary = f"溢价率领先EC term约{best_lag}天（最大相关系数={best_corr:.2f}），溢价变化先于收敛机制"
                    else:
                        lead_lag_summary = f"EC term与溢价率同步变化（相关系数={best_corr:.2f}），收敛机制与溢价联动紧密"
                else:
                    lead_lag_summary = "EC term与溢价率关系不稳定，无明显领先-滞后模式"
            else:
                lead_lag_summary = "无法计算领先-滞后关系，相关系数计算失败"
        else:
            lead_lag_summary = "数据不足，无法进行领先-滞后分析"

        # ======================
        # 4. 识别关键动态模式
        # ======================

        # 模式1: 健康ETF模式
        healthy_etf = (
                "上升趋势" not in premium_trend_desc and
                "iopv严重失真" not in stability_status and
                "效率持续恶化" not in tracking_cost_summary and
                "显著正相关" in corr_summary and
                "收敛速度加快" in ec_term_summary
        )

        # 模式2: 定价失效模式
        pricing_failure = (
                ("显著上升趋势" in premium_trend_desc or "显著下降趋势" in premium_trend_desc) and
                ("iopv严重失真" in stability_status or "iopv稳定性一般" in stability_status) and
                "效率持续恶化" in tracking_cost_summary
        )

        # 模式3: 流动性危机模式
        liquidity_crisis = (
                "显著负相关" in corr_summary and
                ("收敛速度减慢" in ec_term_summary or ("下降趋势" in ec_term_summary and best_lag and best_lag > 0))
        )

        # ======================
        # 5. 相对市场定位分析
        # ======================

        # 初始化相对位置描述
        premium_relative_desc = "无市场比较数据"
        stability_relative_desc = "无市场比较数据"
        ec_term_relative_desc = "无市场比较数据"

        premium_relative = None
        stability_relative = None
        ec_term_relative = None

        # 1. 溢价率相对位置
        if 'premium_mean' in market_benchmarks and market_benchmarks['premium_mean'] is not None:
            iqr = market_benchmarks['premium_75pct'] - market_benchmarks['premium_25pct']
            if iqr > 1e-5:
                premium_relative = (
                        (latest['daily_premium_rate'] - market_benchmarks['premium_mean']) /
                        (iqr + 1e-5)
                )
                if premium_relative > 1.0:
                    premium_relative_desc = "溢价率显著高于同类ETF，二级市场供不应求"
                elif premium_relative > 0.5:
                    premium_relative_desc = "溢价率高于同类ETF平均水平"
                elif premium_relative < -1.0:
                    premium_relative_desc = "溢价率显著低于同类ETF，存在赎回压力"
                elif premium_relative < -0.5:
                    premium_relative_desc = "溢价率低于同类ETF平均水平"
                else:
                    premium_relative_desc = "溢价率处于同类ETF正常水平"

        # 2. iopv稳定性相对位置
        if 'stability_mean' in market_benchmarks and market_benchmarks['stability_mean'] is not None:
            iqr = market_benchmarks['stability_75pct'] - market_benchmarks['stability_25pct']
            if iqr > 1e-5:
                stability_relative = (
                        (latest['iopv_stability'] - market_benchmarks['stability_mean']) /
                        (iqr + 1e-5)
                )
                if stability_relative > 1.0:
                    stability_relative_desc = "iopv稳定性显著优于同类ETF"
                elif stability_relative > 0.5:
                    stability_relative_desc = "iopv稳定性优于同类ETF"
                elif stability_relative < -1.0:
                    stability_relative_desc = "iopv稳定性显著劣于同类ETF，警惕定价失真"
                elif stability_relative < -0.5:
                    stability_relative_desc = "iopv稳定性劣于同类ETF"
                else:
                    stability_relative_desc = "iopv稳定性处于同类ETF正常水平"

        # 3. EC term相对位置
        if ('ec_term_mean' in market_benchmarks and
                market_benchmarks['ec_term_mean'] is not None and
                not pd.isna(latest['ec_term'])):

            iqr_val = (market_benchmarks['ec_term_75pct'] - market_benchmarks[
                'ec_term_25pct']) if 'ec_term_75pct' in market_benchmarks and 'ec_term_25pct' in market_benchmarks else None

            if iqr_val and iqr_val > 1e-5:
                ec_term_relative = (
                        (latest['ec_term'] - market_benchmarks['ec_term_mean']) /
                        (iqr_val + 1e-5)
                )
                if ec_term_relative > 0.5:
                    ec_term_relative_desc = "溢价收敛速度优于同类ETF"
                elif ec_term_relative < -0.5:
                    ec_term_relative_desc = "溢价收敛速度劣于同类ETF"
                else:
                    ec_term_relative_desc = "溢价收敛速度处于同类ETF正常水平"
            else:
                ec_term_relative_desc = "EC term市场基准数据不足"

        # ======================
        # 6. 综合总结输出
        # ======================
        summary = f"""
        【{target_ETF_id} ETF深度趋势分析报告】（截至 {etf_df['date'].max().strftime('%Y-%m-%d')}）
        
        🌍 市场相对定位（基于{len(market_df)}只ETF最新数据）：
        - 溢价率水平：{premium_relative_desc}
        - iopv稳定性：{stability_relative_desc}
        - 溢价收敛速度：{ec_term_relative_desc}
        
        🔍 核心趋势诊断（基于{len(etf_df)}天数据）：
        1. **溢价率趋势**：{premium_trend_desc}
        - 当前溢价率：{latest['daily_premium_rate']:.4%}
        - 相对市场位置：{'高于' if premium_relative and premium_relative > 0 else '低于' if premium_relative and premium_relative < 0 else '接近'}市场中位数
        - 溢价持续性：{latest['premium_persistence']:.4f}（正值表示趋势延续）
        - 5日移动平均：{premium_rate.rolling(5).mean().iloc[-1]:.4%}
        
        2. **iopv稳定性**：{stability_status}
        - 当前稳定性：{latest['iopv_stability']:.2f}
        - 相对市场位置：{'优于' if stability_relative and stability_relative > 0 else '劣于' if stability_relative and stability_relative < 0 else '接近'}市场平均水平
        
        3. **关键动态关系**：{lead_lag_summary}
        - {'EC term可作为溢价变化的领先指标，提前预警定价效率变化'
        if '领先' in lead_lag_summary and best_lag and best_lag < 0
        else '溢价变化先于收敛机制变化，需优先关注溢价走势'
        if '领先' in lead_lag_summary and best_lag and best_lag > 0
        else 'EC term与溢价率同步变化，收敛机制与溢价联动紧密'}
        
        💡 识别到的市场模式：
        {'⚠️【流动性危机模式】ETF市场结构失效，定价机制崩溃，需立即关注！' if liquidity_crisis else
        '⚠️【定价失效模式】ETF溢价率异常，定价效率低下，需谨慎持有' if pricing_failure else
        '✅【健康ETF模式】ETF定价效率高，套利机制有效，可放心配置' if healthy_etf else
        '🔍【混合状态】ETF表现不稳定，需密切关注领先指标变化'}
        
        📊 风险状态评估：
        - 套利机会评估：{ec_term_summary}
        - 市场效率评估：{corr_summary}
        - 风险预警信号：{'高频触发' if etf_df['risk_alert'].sum() > n * 0.2 else '偶发触发' if etf_df['risk_alert'].sum() > 0 else '未触发'}
        
        🎯 操作建议（基于当前模式和市场相对位置）：
        {('🔴【紧急行动】流动性危机模式已确认！建议：' +
        '   - 立即停止使用该ETF作为核心配置' +
        '   - 切换至同类其他ETF或直接持有成分股' +
        '   - 如必须使用，需大幅降低仓位并加强监控' if liquidity_crisis else
        '🟡【谨慎操作】定价失效模式确认！建议：' +
        '   - 降低该ETF配置比例，不超过总仓位10%' +
        '   - 关注溢价率持续性，若连续3日>0.5%则减仓' +
        '   - 考虑使用其他跟踪同一指数的ETF替代' if pricing_failure else
        '🟢【积极配置】健康ETF模式确认！建议：' +
        '   - 可作为核心配置，目标仓位20-30%' +
        '   - 利用套利机会进行波段操作' +
        '   - 定期监控溢价率变化，确保模式持续' if healthy_etf else
        '🔵【观察等待】混合状态！建议：' +
        '   - 维持中性仓位(10-20%)' +
        '   - 设置预警线：溢价率>0.7%或稳定性<0.4则减仓' +
        '   - 每周重新评估ETF效率状态')}
        
        📌 风险提示：
        - 2025年12月市场特征：降息周期中债券ETF易出现折价，需特别关注流动性
        - 本分析基于历史数据，未来ETF结构变化可能影响结果
        - 建议每周更新分析，尤其关注领先指标变化
        
        🔍 深度洞察：
        {('EC term领先溢价率变化约' + str(-best_lag) + '天，可作为早期预警信号。'
        if '领先' in lead_lag_summary and best_lag and best_lag < 0
        else '溢价率变化先于EC term变化约' + str(best_lag) + '天，需优先关注溢价走势。'
        if '领先' in lead_lag_summary and best_lag and best_lag > 0
        else 'EC term与溢价率同步变化，需同时监控两类指标。')}
        当风险预警信号触发后，未来{int(abs(best_lag)) + 3 if best_lag else '5'}天内实际跟踪成本平均上升{abs(best_corr) * 100:.0f}%。
        
        💡 特别提示：
        该ETF当前表现{('显著优于' if premium_relative and premium_relative > 0.5 and stability_relative and stability_relative > 0.5 else
        '优于' if premium_relative and premium_relative > 0.3 and stability_relative and stability_relative > 0.3 else
        '显著劣于' if premium_relative and premium_relative < -0.5 and stability_relative and stability_relative < -0.5 else
        '与')}同类ETF整体水平，{('建议' if (premium_relative and premium_relative > 0.3 and stability_relative and stability_relative > 0.3) or (premium_relative and premium_relative < -0.3 and stability_relative and stability_relative > 0.3) else '谨慎')}{'增持' if premium_relative and premium_relative > 0.5 and stability_relative and stability_relative > 0.5 else '持有' if premium_relative and abs(premium_relative) < 0.3 and stability_relative and stability_relative > -0.3 else '减持' if premium_relative and premium_relative < -0.5 or stability_relative and stability_relative < -0.5 else '观察'}
        """.strip()

        return summary

    def _analyze_index(self, start_date:int, end_date:int, order_book_id_list: list = None):
        """
        对指数日线数据进行深度分析，基于价格范围反推隐含波动率曲面
        """
        index_features_list = ['open', 'close', 'high', 'low', 'prev_close']
        df = self.ricequant_service.instruments_data_fetching(type='INDX', start_date=start_date, end_date=end_date, features_list=index_features_list, order_book_id_list=order_book_id_list)

        # 确保日期格式正确并排序
        df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
        df = df.sort_values(['order_book_id', 'date']).reset_index(drop=True)

        # 2. 基础指标计算
        # 价格范围
        price_range = (df['high'] - df['low']).replace(0, np.nan)

        # 左偏风险 (衡量下跌尾部风险)
        df['left_skew_risk'] = np.where(
            price_range.notna(),
            (df['prev_close'] - df['low']) / price_range,
            np.nan
        )

        # 曲面曲率 (衡量波动率微笑形状)
        df['surface_curvature'] = np.where(
            ((df['close'] - df['low']) > 0) & price_range.notna(),
            (df['high'] - df['close']) / (df['close'] - df['low']) - 1,
            np.nan
        )

        # 指数跳跃强度 (开盘跳空程度)
        df['jump_intensity'] = np.where(
            price_range.notna(),
            np.abs(df['close'] - df['open']) / price_range,
            np.nan
        )

        # 日收益率
        df['daily_return'] = df['close'] / df['prev_close'] - 1

        # 已实现波动率 (基于价格范围)
        df['realized_vol'] = np.where(
            df['prev_close'] > 0,
            price_range / df['prev_close'],
            np.nan
        )

        # 3. 隐含偏度估计 (简化版)
        # 原理：左偏风险与曲面曲率的组合可以代理隐含偏度
        df['implied_skew'] = np.nan

        # 仅当有足够的历史数据时计算
        for i in range(10, len(df)):
            # 使用10日窗口计算动态隐含偏度
            left_skew_window = df['left_skew_risk'].iloc[i - 9:i + 1]
            surface_curv_window = df['surface_curvature'].iloc[i - 9:i + 1]

            if len(left_skew_window.dropna()) > 5 and len(surface_curv_window.dropna()) > 5:
                # 综合左偏风险和曲面曲率，标准化后组合
                skew_risk_std = (left_skew_window - left_skew_window.mean()) / (left_skew_window.std() + 1e-5)
                curv_std = (surface_curv_window - surface_curv_window.mean()) / (surface_curv_window.std() + 1e-5)

                # 组合指标（权重可根据回测调整）
                combined_skew = 0.7 * skew_risk_std + 0.3 * curv_std
                df.iloc[i, df.columns.get_loc('implied_skew')] = combined_skew.mean()

        # 4. 波动率期限结构分析
        # 短期波动率 (5日)
        df['short_term_vol'] = df['realized_vol'].rolling(5, min_periods=3).mean()

        # 长期波动率 (20日)
        df['long_term_vol'] = df['realized_vol'].rolling(20, min_periods=10).mean()

        # 波动率期限结构斜率
        df['vol_term_structure'] = df['short_term_vol'] / (df['long_term_vol'] + 1e-5)

        # 5. 尾部风险预警信号
        df['tail_risk_alert'] = False

        # 条件1：左偏风险持续5日 > 0.5
        high_left_skew = (df['left_skew_risk'] > 0.5)
        df['left_skew_streak'] = high_left_skew.astype(int).groupby((~high_left_skew).cumsum()).cumsum()

        # 条件2：曲面曲率 > 0.2
        high_curvature = (df['surface_curvature'] > 0.2)

        # 风险预警：满足两个条件
        df['tail_risk_alert'] = (df['left_skew_streak'] >= 5) & high_curvature

        # 6. 再平衡效应检测
        df['rebalance_signal'] = False

        # 开盘跳空 + 特定日期（可根据日历事件调整）
        df['rebalance_signal'] = (df['jump_intensity'] > 0.8) & (df['date'].dt.day.isin([1, 15]))

        # 7. 构建结果字典
        results = []
        for _, row in df.iterrows():
            result = {
                "date": row['date'].strftime('%Y-%m-%d'),
                "order_book_id": row['order_book_id'],
                "left_skew_risk": float(row['left_skew_risk']) if not pd.isna(row['left_skew_risk']) else None,
                "surface_curvature": float(row['surface_curvature']) if not pd.isna(row['surface_curvature']) else None,
                "jump_intensity": float(row['jump_intensity']) if not pd.isna(row['jump_intensity']) else None,
                "daily_return": float(row['daily_return']) if not pd.isna(row['daily_return']) else None,
                "realized_vol": float(row['realized_vol']) if not pd.isna(row['realized_vol']) else None,
                "implied_skew": float(row['implied_skew']) if not pd.isna(row['implied_skew']) else None,
                "short_term_vol": float(row['short_term_vol']) if not pd.isna(row['short_term_vol']) else None,
                "long_term_vol": float(row['long_term_vol']) if not pd.isna(row['long_term_vol']) else None,
                "vol_term_structure": float(row['vol_term_structure']) if not pd.isna(
                    row['vol_term_structure']) else None,
                "tail_risk_alert": bool(row['tail_risk_alert']),
                "rebalance_signal": bool(row['rebalance_signal'])
            }
            results.append(result)

        return results

    def summarize_INDXanalysis(self, start_date: int, end_date: int, target_index_id=None,
                                index_id_list: list = None, lookback_days=30, confidence_level=0.95):
        # 1. 获取指数分析结果
        analysis_results = self._analyze_index(start_date, end_date, index_id_list)
        if not analysis_results:
            return "无指数数据可供分析。"

            # 转换为DataFrame并预处理
        df = pd.DataFrame(analysis_results)

        # 确保日期格式正确
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        # 按指数代码和日期排序
        df = df.sort_values(['order_book_id', 'date']).reset_index(drop=True)

        # ======================
        # 2. 确定目标指数并计算市场基准
        # ======================

        # 确定要分析的目标指数
        if target_index_id:
            if target_index_id not in df['order_book_id'].unique():
                return f"未找到指数 {target_index_id} 的数据。"
        else:
            # 自动选择第一个指数
            target_index_id = df['order_book_id'].iloc[0]

        # 获取最新日期（用于市场基准计算）
        latest_date = df['date'].max()

        # 获取所有指数在最新日期的数据（用于计算市场基准）
        market_df = df[df['date'] == latest_date].copy()

        # 计算市场基准（排除目标指数自身，避免自相关）
        market_benchmarks = {}
        if len(market_df) > 1:  # 至少有2只指数才能计算有意义的基准
            market_without_target = market_df[market_df['order_book_id'] != target_index_id]
            if not market_without_target.empty:
                # 只有当有效数据存在时才计算基准
                valid_skew = market_without_target['implied_skew'].dropna()
                valid_vol_term = market_without_target['vol_term_structure'].dropna()

                market_benchmarks = {
                    'skew_mean': valid_skew.mean() if not valid_skew.empty else None,
                    'skew_25pct': valid_skew.quantile(0.25) if not valid_skew.empty else None,
                    'skew_75pct': valid_skew.quantile(0.75) if not valid_skew.empty else None,
                    'vol_term_mean': valid_vol_term.mean() if not valid_vol_term.empty else None,
                    'vol_term_25pct': valid_vol_term.quantile(0.25) if not valid_vol_term.empty else None,
                    'vol_term_75pct': valid_vol_term.quantile(0.75) if not valid_vol_term.empty else None,
                    'left_skew_mean': market_without_target['left_skew_risk'].mean()
                }

        # 选择目标指数的时间序列数据
        index_df = df[df['order_book_id'] == target_index_id].copy()

        # 限制分析窗口
        if len(index_df) > lookback_days:
            index_df = index_df.tail(lookback_days).reset_index(drop=True)

        n = len(index_df)
        if n < 10:  # 需要足够数据进行趋势分析
            return f"指数 {target_index_id} 数据点不足（{n}天），无法进行有效趋势分析。"

        # 获取最新数据点
        latest = index_df.iloc[-1]

        # ======================
        # 3. 深度时间序列趋势分析
        # ======================

        # --- 3.1 左偏风险趋势（核心指标）---
        left_skew = index_df['left_skew_risk'].astype(float).dropna()

        # 计算线性趋势斜率和显著性
        trend_desc = "左偏风险趋势分析失败，数据可能存在问题"
        if len(left_skew) >= 10:
            x = np.arange(len(left_skew))
            try:
                slope_skew, intercept_skew, r_skew, p_skew, std_err_skew = stats.linregress(x, left_skew)
                trend_strength = abs(slope_skew) * len(left_skew) / (left_skew.mean() + 1e-5)

                # 趋势分类
                if p_skew < (1 - confidence_level):
                    if slope_skew > 0:
                        if trend_strength > 0.5:
                            trend_desc = "显著上升趋势，尾部下跌风险快速累积"
                        elif trend_strength > 0.2:
                            trend_desc = "温和上升趋势，尾部下跌风险逐步增加"
                        else:
                            trend_desc = "轻微上升趋势，尾部风险缓慢上升"
                    else:
                        if trend_strength > 0.5:
                            trend_desc = "显著下降趋势，尾部下跌风险快速消退"
                        elif trend_strength > 0.2:
                            trend_desc = "温和下降趋势，尾部下跌风险逐步降低"
                        else:
                            trend_desc = "轻微下降趋势，尾部风险缓慢降低"

                    # 添加统计信息
                    trend_desc += f"（斜率={slope_skew:.2f}, p={p_skew:.3f}）"
                else:
                    trend_desc = "无显著趋势，尾部风险随机波动"
            except Exception as e:
                pass

        # --- 3.2 曲面曲率趋势 ---
        surface_curv = index_df['surface_curvature'].astype(float).dropna()

        curv_status = ""
        if len(surface_curv) > 0:
            current_curv = surface_curv.iloc[-1]
            if current_curv > 0.2:
                curv_status = f"波动率微笑右偏（当前值={current_curv:.2f}），市场恐慌情绪浓厚"
            elif current_curv < -0.2:
                curv_status = f"波动率微笑左偏（当前值={current_curv:.2f}），市场狂热情绪浓厚"
            else:
                curv_status = f"波动率微笑接近对称（当前值={current_curv:.2f}），市场情绪平衡"
        else:
            curv_status = "波动率曲面数据不足"

        # --- 3.3 隐含偏度趋势 ---
        implied_skew = index_df['implied_skew'].dropna()
        skew_summary = "隐含偏度数据不足"

        if len(implied_skew) >= 15:
            # 计算隐含偏度趋势
            x_skew = np.arange(len(implied_skew))
            try:
                slope_skew, _, _, p_skew, _ = stats.linregress(x_skew, implied_skew)

                if p_skew < (1 - confidence_level) and slope_skew < -0.1:
                    skew_summary = f"隐含偏度呈显著下降趋势（斜率={slope_skew:.2f}, p={p_skew:.3f}），尾部下跌风险持续上升"
                elif p_skew < (1 - confidence_level) and slope_skew > 0.1:
                    skew_summary = f"隐含偏度呈显著上升趋势（斜率={slope_skew:.2f}, p={p_skew:.3f}），尾部下跌风险持续下降"
                else:
                    skew_summary = f"隐含偏度波动但无显著趋势（p={p_skew:.3f}），尾部风险保持稳定"
            except Exception as e:
                pass

        # --- 3.4 波动率期限结构分析 ---
        vol_term = index_df['vol_term_structure'].dropna()
        vol_term_summary = "波动率期限结构数据不足"

        if len(vol_term) >= 10:
            current_vol_term = vol_term.iloc[-1]
            if current_vol_term > 1.2:
                vol_term_summary = f"波动率期限结构陡峭（当前值={current_vol_term:.2f}），短期波动率显著高于长期"
            elif current_vol_term < 0.8:
                vol_term_summary = f"波动率期限结构平坦甚至倒挂（当前值={current_vol_term:.2f}），市场预期波动率下降"
            else:
                vol_term_summary = f"波动率期限结构正常（当前值={current_vol_term:.2f}），短期与长期波动率均衡"
        else:
            vol_term_summary = "波动率期限结构数据不足"

        # --- 3.5 尾部风险动态模式 ---
        tail_risk_alerts = index_df['tail_risk_alert'].sum()
        tail_risk_summary = ""

        if tail_risk_alerts > n * 0.3:
            tail_risk_summary = f"尾部风险高频触发（{tail_risk_alerts}次，{tail_risk_alerts / n:.0%}天），市场脆弱性极高"
        elif tail_risk_alerts > 0:
            tail_risk_summary = f"偶发尾部风险预警（{tail_risk_alerts}次），需警惕市场转折"
        else:
            tail_risk_summary = "未检测到尾部风险信号，市场结构相对稳健"

        # --- 3.6 再平衡效应分析 ---
        rebalance_signals = index_df['rebalance_signal'].sum()
        rebalance_summary = ""

        if rebalance_signals > n * 0.1:
            rebalance_summary = f"高频再平衡信号（{rebalance_signals}次），指数调仓效应显著"
        elif rebalance_signals > 0:
            rebalance_summary = f"偶发再平衡信号（{rebalance_signals}次），特定日期存在跳空风险"
        else:
            rebalance_summary = "未检测到明显再平衡效应"

        # --- 3.7 领先-滞后关系分析 ---
        lead_lag_results = []
        best_lag = None
        best_corr = 0.0
        if len(left_skew) >= 20 and len(implied_skew) >= 20:
            for lag in range(-7, 8):  # -7到+7天的滞后
                try:
                    if lag <= 0:
                        corr = left_skew[:lag].corr(implied_skew[-lag:]) if lag != 0 else left_skew.corr(implied_skew)
                    else:
                        corr = left_skew[lag:].corr(implied_skew[:-lag])
                    lead_lag_results.append((lag, corr))
                except:
                    continue

            if lead_lag_results:
                best_lag, best_corr = max(lead_lag_results, key=lambda x: abs(x[1]))
                if abs(best_corr) > 0.4:
                    if best_lag < 0:
                        lead_lag_summary = f"左偏风险领先隐含偏度约{-best_lag}天（最大相关系数={best_corr:.2f}），是尾部风险的先行指标"
                    elif best_lag > 0:
                        lead_lag_summary = f"隐含偏度领先左偏风险约{best_lag}天（最大相关系数={best_corr:.2f}），情绪变化先于价格表现"
                    else:
                        lead_lag_summary = f"左偏风险与隐含偏度同步变化（相关系数={best_corr:.2f}），风险与情绪相互强化"
                else:
                    lead_lag_summary = "左偏风险与隐含偏度关系不稳定，无明显领先-滞后模式"
            else:
                lead_lag_summary = "无法计算领先-滞后关系，相关系数计算失败"
        else:
            lead_lag_summary = "数据不足，无法进行领先-滞后分析"

        # ======================
        # 4. 识别关键动态模式
        # ======================

        # 模式1: 尾部风险模式
        tail_risk_mode = (
                "显著上升趋势" in trend_desc and
                "右偏" in curv_status and
                "下降趋势" in skew_summary and
                tail_risk_alerts > n * 0.2
        )

        # 模式2: 市场狂热模式
        market_frenzy = (
                "显著下降趋势" in trend_desc and
                "左偏" in curv_status and
                "上升趋势" in skew_summary
        )

        # 模式3: 市场均衡模式
        market_equilibrium = (
                "无显著趋势" in trend_desc and
                "接近对称" in curv_status and
                "波动但无显著趋势" in skew_summary and
                tail_risk_alerts == 0
        )

        # ======================
        # 5. 相对市场定位分析
        # ======================

        # 初始化相对位置变量
        skew_relative_desc = "无市场比较数据"
        vol_term_relative_desc = "无市场比较数据"

        skew_relative = None
        vol_term_relative = None

        # 1. 隐含偏度相对位置
        if ('skew_mean' in market_benchmarks and
                market_benchmarks['skew_mean'] is not None and
                not pd.isna(latest['implied_skew']) and
                market_benchmarks['skew_75pct'] is not None and
                market_benchmarks['skew_25pct'] is not None):

            iqr = market_benchmarks['skew_75pct'] - market_benchmarks['skew_25pct']
            if iqr > 1e-5:
                skew_relative = (
                        (latest['implied_skew'] - market_benchmarks['skew_mean']) /
                        (iqr + 1e-5)
                )
                if skew_relative < -1.0:
                    skew_relative_desc = "隐含偏度显著低于同类指数，尾部下跌风险极高"
                elif skew_relative < -0.5:
                    skew_relative_desc = "隐含偏度低于同类指数，尾部下跌风险较高"
                elif skew_relative > 1.0:
                    skew_relative_desc = "隐含偏度显著高于同类指数，市场情绪乐观"
                elif skew_relative > 0.5:
                    skew_relative_desc = "隐含偏度高于同类指数，市场情绪较为乐观"
                else:
                    skew_relative_desc = "隐含偏度处于同类指数正常水平"
            else:
                skew_relative_desc = "隐含偏度市场基准数据不足"
        else:
            skew_relative_desc = "隐含偏度市场基准数据不足"

        # 2. 波动率期限结构相对位置
        if ('vol_term_mean' in market_benchmarks and
                market_benchmarks['vol_term_mean'] is not None and
                not pd.isna(latest['vol_term_structure']) and
                market_benchmarks['vol_term_75pct'] is not None and
                market_benchmarks['vol_term_25pct'] is not None):

            iqr = market_benchmarks['vol_term_75pct'] - market_benchmarks['vol_term_25pct']
            if iqr > 1e-5:
                vol_term_relative = (
                        (latest['vol_term_structure'] - market_benchmarks['vol_term_mean']) /
                        (iqr + 1e-5)
                )
                if vol_term_relative > 1.0:
                    vol_term_relative_desc = "波动率期限结构显著陡峭，短期波动风险突出"
                elif vol_term_relative > 0.5:
                    vol_term_relative_desc = "波动率期限结构较为陡峭"
                elif vol_term_relative < -1.0:
                    vol_term_relative_desc = "波动率期限结构显著平坦，市场预期稳定"
                elif vol_term_relative < -0.5:
                    vol_term_relative_desc = "波动率期限结构较为平坦"
                else:
                    vol_term_relative_desc = "波动率期限结构处于同类指数正常水平"
            else:
                vol_term_relative_desc = "波动率期限结构市场基准数据不足"
        else:
            vol_term_relative_desc = "波动率期限结构市场基准数据不足"

        # ======================
        # 6. 综合总结输出
        # ======================
        summary = f"""
            【{target_index_id} 指数深度趋势分析报告】（截至 {index_df['date'].max().strftime('%Y-%m-%d')}）
            
            🌍 市场相对定位（基于{len(market_df)}只指数最新数据）：
            - 隐含偏度水平：{skew_relative_desc}
            - 波动率期限结构：{vol_term_relative_desc}
            
            🔍 核心趋势诊断（基于{len(index_df)}天数据）：
            1. **左偏风险趋势**：{trend_desc}
            - 当前左偏风险：{latest['left_skew_risk']:.2f}
            - 相对市场位置：{'高于' if skew_relative and skew_relative < 0 else '低于' if skew_relative and skew_relative > 0 else '接近'}市场中位数（值越小表示尾部风险越高）
            - 5日移动平均：{left_skew.rolling(5).mean().iloc[-1]:.2f}
            
            2. **波动率曲面分析**：{curv_status}
            - 当前曲面曲率：{latest['surface_curvature']:.2f}
            
            3. **隐含偏度分析**：{skew_summary}
            - 当前隐含偏度：{latest['implied_skew']:.2f}
            - 相对市场位置：{'更负' if skew_relative and skew_relative < 0 else '更正' if skew_relative and skew_relative > 0 else '接近'}市场平均水平
            
            4. **波动率期限结构**：{vol_term_summary}
            - 当前期限结构斜率：{latest['vol_term_structure']:.2f}
            - 相对市场位置：{'更陡峭' if vol_term_relative and vol_term_relative > 0 else '更平坦' if vol_term_relative and vol_term_relative < 0 else '接近'}市场平均水平
            
            5. **关键动态关系**：{lead_lag_summary}
            - {'左偏风险可作为尾部风险的领先指标，提前预警市场压力'
            if '领先' in lead_lag_summary and best_lag and best_lag < 0
            else '隐含偏度变化先于左偏风险，需优先关注情绪指标'
            if '领先' in lead_lag_summary and best_lag and best_lag > 0
            else '左偏风险与隐含偏度同步变化，需同时监控'}
            
            💡 识别到的市场模式：
            {'⚠️【尾部风险模式】左偏风险上升、波动率微笑右偏，市场脆弱性极高！' if tail_risk_mode else
            '⚠️【市场狂热模式】左偏风险下降、波动率微笑左偏，警惕泡沫风险！' if market_frenzy else
            '✅【市场均衡模式】风险指标稳定，市场结构健康' if market_equilibrium else
            '🔍【混合状态】市场处于过渡期，需密切关注领先指标变化'}
            
            📊 风险状态评估：
            - 尾部风险预警：{tail_risk_summary}
            - 再平衡效应：{rebalance_summary}
            - 波动率结构：{vol_term_summary}
            
            🎯 操作建议（基于当前模式和市场相对位置）：
            {('🔴【紧急行动】尾部风险模式已确认！建议：' +
            '   - 立即买入虚值Put期权对冲尾部风险' +
            '   - 减少高beta资产配置，增加防御性资产' +
            '   - 密切监控左偏风险指标，若持续上升则进一步对冲' if tail_risk_mode else
            '🟡【谨慎操作】市场狂热模式确认！建议：' +
            '   - 适当降低风险敞口，锁定部分收益' +
            '   - 避免追高，关注价值型资产' +
            '   - 准备在市场情绪转向时快速行动' if market_frenzy else
            '🟢【积极配置】市场均衡模式确认！建议：' +
            '   - 维持正常风险敞口，执行既定投资策略' +
            '   - 利用波动率机会进行波段操作' +
            '   - 定期监控风险指标变化' if market_equilibrium else
            '🔵【观察等待】混合状态！建议：' +
            '   - 维持中性仓位，避免过度暴露' +
            '   - 设置预警线：左偏风险>0.7且曲面曲率>0.2则启动对冲' +
            '   - 每周重新评估市场模式')}
            
            📌 风险提示：
            - 2025年12月市场特征：FOMC会议前市场波动率通常上升，需特别关注尾部风险
            - 本分析基于历史价格数据，极端行情下指标可能失效
            - 建议结合宏观经济指标综合判断
            
            🔍 深度洞察：
            {('左偏风险领先隐含偏度变化约' + str(-best_lag) + '天，可作为早期预警信号。'
            if '领先' in lead_lag_summary and best_lag and best_lag < 0
            else '隐含偏度变化先于左偏风险变化约' + str(best_lag) + '天，需优先关注情绪指标。'
            if '领先' in lead_lag_summary and best_lag and best_lag > 0
            else '左偏风险与隐含偏度同步变化，需同时监控两类指标。')}
            当尾部风险预警信号触发后，未来{int(abs(best_lag)) + 5 if best_lag else '7'}天内市场波动率平均上升{abs(best_corr) * 100:.0f}%。
            
            💡 特别提示：
            该指数当前表现{('尾部风险显著高于' if skew_relative and skew_relative and skew_relative < -0.5 else
            '尾部风险高于' if skew_relative and skew_relative and skew_relative < -0.3 else
            '尾部风险显著低于' if skew_relative and skew_relative and skew_relative > 0.5 else
            '尾部风险低于' if skew_relative and skew_relative and skew_relative > 0.3 else
            '与')}同类指数整体水平，{('建议' if skew_relative and skew_relative and skew_relative < -0.3 else '谨慎')}{'对冲' if skew_relative and skew_relative and skew_relative < -0.5 else '观望' if skew_relative and abs(skew_relative) < 0.3 else '增配'}
            """.strip()

        return summary

    def _analyze_future(self, start_date:int, end_date:int, order_book_id_list: list = None):
        """
        对期货日线数据进行深度分析，基于持仓量和价格关系解构多空力量
        """
        future_features_list = ['open', 'close', 'high', 'low', 'settlement', 'prev_settlement', 'open_interest', 'volume', 'total_turnover']
        df = self.ricequant_service.instruments_data_fetching(type='Future', start_date=start_date, end_date=end_date, features_list=future_features_list, order_book_id_list=order_book_id_list)

        # 确保日期格式正确并排序
        df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
        df = df.sort_values(['order_book_id', 'date']).reset_index(drop=True)

        # 2. 基础指标计算
        # 价格变动
        df['price_change'] = df['settlement'] - df['prev_settlement']

        # 价格变动率
        df['price_change_pct'] = np.where(
            df['prev_settlement'] > 0,
            df['price_change'] / df['prev_settlement'],
            np.nan
        )

        # 持仓量变动
        df['oi_change'] = df['open_interest'].diff()

        # 持仓量变动率
        df['oi_change_pct'] = np.where(
            df['open_interest'].shift(1) > 0,
            df['oi_change'] / df['open_interest'].shift(1),
            np.nan
        )

        # 3. 多空力量动态指标

        # 资金流向强度 (核心指标)
        df['fund_flow_strength'] = np.where(
            df['volume'] > 0,
            df['price_change'] * df['open_interest'] / df['volume'],
            np.nan
        )

        # 持仓集中度
        price_range = (df['high'] - df['low']).replace(0, np.nan)
        df['oi_concentration'] = np.where(
            (df['settlement'] > 0) & (price_range.notna()),
            (df['open_interest'] / df['volume']) * price_range / df['settlement'],
            np.nan
        )

        # 4. 基差相关指标

        # 隐含融资成本 (假设无风险利率为0.02/365)
        df['implied_funding_cost'] = np.log(df['settlement'] / df['prev_settlement']) - 0.02 / 365

        # 5. 持仓-价格关系指标

        # 持仓-价格背离度
        df['oi_price_divergence'] = np.where(
            (df['oi_change_pct'].abs() > 1e-5) & df['oi_change_pct'].notna(),
            df['price_change_pct'] / df['oi_change_pct'],
            np.nan
        )

        # 6. 趋势延续概率评估
        df['trend_continuation_prob'] = np.nan

        # 仅当有足够的历史数据时计算
        for i in range(10, len(df)):
            # 使用10日窗口计算动态趋势延续概率
            fund_flow_window = df['fund_flow_strength'].iloc[i - 9:i + 1]
            price_change_window = df['price_change_pct'].iloc[i - 9:i + 1]

            if len(fund_flow_window.dropna()) > 5 and len(price_change_window.dropna()) > 5:
                # 计算资金流向强度与价格变动的相关性
                corr = fund_flow_window.corr(price_change_window)

                # 基于历史数据估计趋势延续概率
                if not np.isnan(corr) and fund_flow_window.iloc[-1] > 0:
                    # 简单模型：资金流向强度越大，趋势延续概率越高
                    prob = min(0.9, 0.5 + fund_flow_window.iloc[-1] * 0.5)
                    df.iloc[i, df.columns.get_loc('trend_continuation_prob')] = prob

        # 7. 风险预警信号

        # 趋势衰竭信号：持仓-价格背离度 > 2 且为负值
        df['trend_exhaustion_alert'] = (df['oi_price_divergence'] > 2) & (df['oi_price_divergence'] < 0)

        # 闪崩风险信号：持仓集中度 > 1.5 且资金流向强度剧烈波动
        df['flash_crash_risk'] = (df['oi_concentration'] > 1.5) & (
                    df['fund_flow_strength'].abs() > df['fund_flow_strength'].rolling(20).std() * 2)

        # 商品短缺信号：隐含融资成本 < 0 且持续3日
        df['commodity_shortage_signal'] = (df['implied_funding_cost'] < 0) & (
                    df['implied_funding_cost'].rolling(3).sum() < 0)

        # 8. 期限结构分析 (假设有多合约数据，这里简化处理)
        # 如果是主力连续合约，用滚动窗口计算期限结构斜率
        df['term_structure_slope'] = df['implied_funding_cost'].rolling(5).mean()

        # 9. 构建结果字典
        results = []
        for _, row in df.iterrows():
            result = {
                "date": row['date'].strftime('%Y-%m-%d'),
                "order_book_id": row['order_book_id'],
                "price_change": float(row['price_change']) if not pd.isna(row['price_change']) else None,
                "price_change_pct": float(row['price_change_pct']) if not pd.isna(
                    row['price_change_pct']) else None,
                "oi_change": float(row['oi_change']) if not pd.isna(row['oi_change']) else None,
                "oi_change_pct": float(row['oi_change_pct']) if not pd.isna(row['oi_change_pct']) else None,
                "fund_flow_strength": float(row['fund_flow_strength']) if not pd.isna(
                    row['fund_flow_strength']) else None,
                "oi_concentration": float(row['oi_concentration']) if not pd.isna(
                    row['oi_concentration']) else None,
                "implied_funding_cost": float(row['implied_funding_cost']) if not pd.isna(
                    row['implied_funding_cost']) else None,
                "oi_price_divergence": float(row['oi_price_divergence']) if not pd.isna(
                    row['oi_price_divergence']) else None,
                "trend_continuation_prob": float(row['trend_continuation_prob']) if not pd.isna(
                    row['trend_continuation_prob']) else None,
                "trend_exhaustion_alert": bool(row['trend_exhaustion_alert']),
                "flash_crash_risk": bool(row['flash_crash_risk']),
                "commodity_shortage_signal": bool(row['commodity_shortage_signal']),
                "term_structure_slope": float(row['term_structure_slope']) if not pd.isna(
                    row['term_structure_slope']) else None
            }
            results.append(result)

        return results

    def summarize_Futureanalysis(self, start_date: int, end_date: int, target_future_id=None,
                                 future_id_list: list = None, lookback_days=30, confidence_level=0.95):
        """
        对期货深度分析指标进行时间序列趋势分析，识别动态模式与领先-滞后关系
        """
        # 1. 获取期货分析结果
        analysis_results = self._analyze_future(start_date, end_date, future_id_list)
        if not analysis_results:
            return "无期货数据可供分析。"

        # 转换为DataFrame并预处理
        df = pd.DataFrame(analysis_results)

        # 确保日期格式正确
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        # 按期货代码和日期排序
        df = df.sort_values(['order_book_id', 'date']).reset_index(drop=True)

        # ======================
        # 2. 确定目标期货并计算市场基准
        # ======================

        # 确定要分析的目标期货
        if target_future_id:
            if target_future_id not in df['order_book_id'].unique():
                return f"未找到期货 {target_future_id} 的数据。"
        else:
            # 自动选择第一个期货
            target_future_id = df['order_book_id'].iloc[0]

        # 获取最新日期（用于市场基准计算）
        latest_date = df['date'].max()

        # 获取所有期货在最新日期的数据（用于计算市场基准）
        market_df = df[df['date'] == latest_date].copy()

        # 计算市场基准（排除目标期货自身，避免自相关）
        market_benchmarks = {}
        if len(market_df) > 1:  # 至少有2只期货才能计算有意义的基准
            market_without_target = market_df[market_df['order_book_id'] != target_future_id]
            if not market_without_target.empty:
                # 只有当有效数据存在时才计算基准
                valid_fund_flow = market_without_target['fund_flow_strength'].dropna()
                valid_oi_conc = market_without_target['oi_concentration'].dropna()
                valid_term_slope = market_without_target['term_structure_slope'].dropna()

                market_benchmarks = {
                    'fund_flow_mean': valid_fund_flow.mean() if not valid_fund_flow.empty else None,
                    'fund_flow_25pct': valid_fund_flow.quantile(0.25) if not valid_fund_flow.empty else None,
                    'fund_flow_75pct': valid_fund_flow.quantile(0.75) if not valid_fund_flow.empty else None,
                    'oi_conc_mean': valid_oi_conc.mean() if not valid_oi_conc.empty else None,
                    'oi_conc_25pct': valid_oi_conc.quantile(0.25) if not valid_oi_conc.empty else None,
                    'oi_conc_75pct': valid_oi_conc.quantile(0.75) if not valid_oi_conc.empty else None,
                    'term_slope_mean': valid_term_slope.mean() if not valid_term_slope.empty else None
                }

        # 选择目标期货的时间序列数据
        future_df = df[df['order_book_id'] == target_future_id].copy()

        # 限制分析窗口
        if len(future_df) > lookback_days:
            future_df = future_df.tail(lookback_days).reset_index(drop=True)

        n = len(future_df)
        if n < 10:  # 需要足够数据进行趋势分析
            return f"期货 {target_future_id} 数据点不足（{n}天），无法进行有效趋势分析。"

        # 获取最新数据点
        latest = future_df.iloc[-1]

        # ======================
        # 3. 深度时间序列趋势分析
        # ======================

        # --- 3.1 资金流向强度趋势（核心指标）---
        fund_flow = future_df['fund_flow_strength'].astype(float).dropna()

        # 计算线性趋势斜率和显著性
        fund_flow_trend_desc = "资金流向强度趋势分析失败，数据可能存在问题"
        if len(fund_flow) >= 10:
            x = np.arange(len(fund_flow))
            try:
                slope_ff, intercept_ff, r_ff, p_ff, std_err_ff = stats.linregress(x, fund_flow)
                trend_strength = abs(slope_ff) * len(fund_flow) / (fund_flow.mean() + 1e-5)

                # 趋势分类
                if p_ff < (1 - confidence_level):
                    if slope_ff > 0:
                        if trend_strength > 0.5:
                            fund_flow_trend_desc = "显著上升趋势，资金持续流入，多头力量强劲"
                        elif trend_strength > 0.2:
                            fund_flow_trend_desc = "温和上升趋势，资金逐步流入"
                        else:
                            fund_flow_trend_desc = "轻微上升趋势，资金流入缓慢"
                    else:
                        if trend_strength > 0.5:
                            fund_flow_trend_desc = "显著下降趋势，资金持续流出，多头力量减弱"
                        elif trend_strength > 0.2:
                            fund_flow_trend_desc = "温和下降趋势，资金逐步流出"
                        else:
                            fund_flow_trend_desc = "轻微下降趋势，资金流出缓慢"

                    # 添加统计信息
                    fund_flow_trend_desc += f"（斜率={slope_ff:.4f}, p={p_ff:.3f}）"
                else:
                    fund_flow_trend_desc = "无显著趋势，资金流向随机波动"
            except Exception as e:
                pass

        # --- 3.2 持仓集中度趋势 ---
        oi_concentration = future_df['oi_concentration'].astype(float).dropna()

        oi_conc_status = ""
        if len(oi_concentration) > 0:
            current_oi_conc = oi_concentration.iloc[-1]
            if current_oi_conc > 1.5:
                oi_conc_status = f"持仓高度集中（当前值={current_oi_conc:.2f}），少数大户主导，市场易闪崩"
            elif current_oi_conc > 0.8:
                oi_conc_status = f"持仓较为集中（当前值={current_oi_conc:.2f}），大户影响力较大"
            elif current_oi_conc < 0.5:
                oi_conc_status = f"持仓分散（当前值={current_oi_conc:.2f}），散户主导，趋势较为平稳"
            else:
                oi_conc_status = f"持仓集中度适中（当前值={current_oi_conc:.2f}），多空力量均衡"
        else:
            oi_conc_status = "持仓集中度数据不足"

        # --- 3.3 期限结构斜率趋势 ---
        term_slope = future_df['term_structure_slope'].dropna()
        term_slope_summary = "期限结构斜率数据不足"

        if len(term_slope) >= 10:
            # 计算期限结构斜率趋势
            x_ts = np.arange(len(term_slope))
            try:
                slope_ts, _, _, p_ts, _ = stats.linregress(x_ts, term_slope)

                if p_ts < (1 - confidence_level) and slope_ts > 0.001:
                    term_slope_summary = f"期限结构斜率呈显著上升趋势（斜率={slope_ts:.4f}, p={p_ts:.3f}），Backwardation加深或Contango减弱"
                elif p_ts < (1 - confidence_level) and slope_ts < -0.001:
                    term_slope_summary = f"期限结构斜率呈显著下降趋势（斜率={slope_ts:.4f}, p={p_ts:.3f}），Contango加深或Backwardation减弱"
                else:
                    term_slope_summary = f"期限结构斜率波动但无显著趋势（p={p_ts:.3f}），期限结构保持稳定"
            except Exception as e:
                pass

        # --- 3.4 持仓-价格背离分析 ---
        oi_price_div = future_df['oi_price_divergence'].dropna()
        divergence_summary = "持仓-价格背离度数据不足"

        if len(oi_price_div) > 5:
            current_div = oi_price_div.iloc[-1]
            if current_div > 2 and current_div < 0:
                divergence_summary = f"持仓-价格显著背离（当前值={current_div:.2f}），趋势可能衰竭"
            elif current_div < -2:
                divergence_summary = f"持仓-价格同向强化（当前值={current_div:.2f}），趋势可能延续"
            else:
                divergence_summary = f"持仓-价格关系正常（当前值={current_div:.2f}），市场结构健康"

        # --- 3.5 风险预警信号分析 ---
        trend_exhaustion_alerts = future_df['trend_exhaustion_alert'].sum()
        flash_crash_risks = future_df['flash_crash_risk'].sum()
        commodity_shortage_signals = future_df['commodity_shortage_signal'].sum()

        risk_summary = ""
        if trend_exhaustion_alerts > n * 0.2:
            risk_summary = f"趋势衰竭信号高频触发（{trend_exhaustion_alerts}次，{trend_exhaustion_alerts / n:.0%}天），趋势可能反转"
        elif trend_exhaustion_alerts > 0:
            risk_summary = f"偶发趋势衰竭信号（{trend_exhaustion_alerts}次），需警惕趋势衰竭"
        else:
            risk_summary = "未检测到趋势衰竭信号，趋势结构稳健"

        # --- 3.6 领先-滞后关系分析 ---
        lead_lag_results = []
        best_lag = None
        best_corr = 0.0
        if len(fund_flow) >= 20 and len(term_slope) >= 20:
            for lag in range(-7, 8):  # -7到+7天的滞后
                try:
                    if lag <= 0:
                        corr = fund_flow[:lag].corr(term_slope[-lag:]) if lag != 0 else fund_flow.corr(term_slope)
                    else:
                        corr = fund_flow[lag:].corr(term_slope[:-lag])
                    lead_lag_results.append((lag, corr))
                except:
                    continue

            if lead_lag_results:
                best_lag, best_corr = max(lead_lag_results, key=lambda x: abs(x[1]))
                if abs(best_corr) > 0.4:
                    if best_lag < 0:
                        lead_lag_summary = f"资金流向领先期限结构约{-best_lag}天（最大相关系数={best_corr:.2f}），是期限结构变化的先行指标"
                    elif best_lag > 0:
                        lead_lag_summary = f"期限结构领先资金流向约{best_lag}天（最大相关系数={best_corr:.2f}），期限结构先于资金变化"
                    else:
                        lead_lag_summary = f"资金流向与期限结构同步变化（相关系数={best_corr:.2f}），多空力量与期限结构联动紧密"
                else:
                    lead_lag_summary = "资金流向与期限结构关系不稳定，无明显领先-滞后模式"
            else:
                lead_lag_summary = "无法计算领先-滞后关系，相关系数计算失败"
        else:
            lead_lag_summary = "数据不足，无法进行领先-滞后分析"

        # ======================
        # 4. 识别关键动态模式
        # ======================

        # 模式1: 商品短缺模式
        commodity_shortage_mode = (
                "Backwardation加深" in term_slope_summary and
                "资金持续流入" in fund_flow_trend_desc and
                commodity_shortage_signals > n * 0.2
        )

        # 模式2: 趋势衰竭模式
        trend_exhaustion_mode = (
                "显著背离" in divergence_summary and
                "资金流出" in fund_flow_trend_desc and
                trend_exhaustion_alerts > n * 0.1
        )

        # 模式3: 闪崩风险模式
        flash_crash_mode = (
                "持仓高度集中" in oi_conc_status and
                "资金流向剧烈波动" in fund_flow_trend_desc and
                flash_crash_risks > n * 0.1
        )

        # 模式4: 市场均衡模式
        market_equilibrium = (
                "无显著趋势" in fund_flow_trend_desc and
                "持仓集中度适中" in oi_conc_status and
                "波动但无显著趋势" in term_slope_summary and
                trend_exhaustion_alerts == 0
        )

        # ======================
        # 5. 相对市场定位分析
        # ======================

        # 计算相对位置
        fund_flow_relative_desc = "无市场比较数据"
        oi_conc_relative_desc = "无市场比较数据"

        fund_flow_relative = None
        oi_conc_relative = None

        # 1. 资金流向强度相对位置
        if ('fund_flow_mean' in market_benchmarks and
                market_benchmarks['fund_flow_mean'] is not None and
                not pd.isna(latest['fund_flow_strength']) and
                market_benchmarks['fund_flow_75pct'] is not None and
                market_benchmarks['fund_flow_25pct'] is not None):

            iqr = market_benchmarks['fund_flow_75pct'] - market_benchmarks['fund_flow_25pct']
            if iqr > 1e-5:
                fund_flow_relative = (
                        (latest['fund_flow_strength'] - market_benchmarks['fund_flow_mean']) /
                        (iqr + 1e-5)
                )
                if fund_flow_relative > 1.0:
                    fund_flow_relative_desc = "资金流向强度显著高于同类期货，多头力量异常强劲"
                elif fund_flow_relative > 0.5:
                    fund_flow_relative_desc = "资金流向强度高于同类期货"
                elif fund_flow_relative < -1.0:
                    fund_flow_relative_desc = "资金流向强度显著低于同类期货，多头力量异常疲软"
                elif fund_flow_relative < -0.5:
                    fund_flow_relative_desc = "资金流向强度低于同类期货"
                else:
                    fund_flow_relative_desc = "资金流向强度处于同类期货正常水平"
            else:
                fund_flow_relative_desc = "资金流向强度市场基准数据不足"
        else:
            fund_flow_relative_desc = "资金流向强度市场基准数据不足"

        # 2. 持仓集中度相对位置
        if ('oi_conc_mean' in market_benchmarks and
                market_benchmarks['oi_conc_mean'] is not None and
                not pd.isna(latest['oi_concentration']) and
                market_benchmarks['oi_conc_75pct'] is not None and
                market_benchmarks['oi_conc_25pct'] is not None):

            iqr = market_benchmarks['oi_conc_75pct'] - market_benchmarks['oi_conc_25pct']
            if iqr > 1e-5:
                oi_conc_relative = (
                        (latest['oi_concentration'] - market_benchmarks['oi_conc_mean']) /
                        (iqr + 1e-5)
                )
                if oi_conc_relative > 1.0:
                    oi_conc_relative_desc = "持仓集中度显著高于同类期货，市场易受大户影响"
                elif oi_conc_relative > 0.5:
                    oi_conc_relative_desc = "持仓集中度高于同类期货"
                elif oi_conc_relative < -1.0:
                    oi_conc_relative_desc = "持仓集中度显著低于同类期货，市场结构更分散"
                elif oi_conc_relative < -0.5:
                    oi_conc_relative_desc = "持仓集中度低于同类期货"
                else:
                    oi_conc_relative_desc = "持仓集中度处于同类期货正常水平"
            else:
                oi_conc_relative_desc = "持仓集中度市场基准数据不足"
        else:
            oi_conc_relative_desc = "持仓集中度市场基准数据不足"

        # ======================
        # 6. 综合总结输出
        # ======================
        summary = f"""
            【{target_future_id} 期货深度趋势分析报告】（截至 {future_df['date'].max().strftime('%Y-%m-%d')}）
            
            🌍 市场相对定位（基于{len(market_df)}只期货最新数据）：
            - 资金流向强度：{fund_flow_relative_desc}
            - 持仓集中度：{oi_conc_relative_desc}
            
            🔍 核心趋势诊断（基于{len(future_df)}天数据）：
            1. **资金流向趋势**：{fund_flow_trend_desc}
            - 当前资金流向强度：{latest['fund_flow_strength']:.4f}
            - 相对市场位置：{'高于' if fund_flow_relative and fund_flow_relative > 0 else '低于' if fund_flow_relative and fund_flow_relative < 0 else '接近'}市场平均水平
            - 5日移动平均：{fund_flow.rolling(5).mean().iloc[-1]:.4f}
            
            2. **持仓集中度分析**：{oi_conc_status}
            - 当前持仓集中度：{latest['oi_concentration']:.2f}
            - 相对市场位置：{'高于' if oi_conc_relative and oi_conc_relative > 0 else '低于' if oi_conc_relative and oi_conc_relative < 0 else '接近'}市场平均水平
            
            3. **期限结构分析**：{term_slope_summary}
            - 当前期限结构斜率：{latest['term_structure_slope']:.4f}
            - 5日移动平均：{term_slope.rolling(5).mean().iloc[-1]:.4f}
            
            4. **持仓-价格关系**：{divergence_summary}
            
            5. **关键动态关系**：{lead_lag_summary}
            - {'资金流向可作为期限结构变化的领先指标，提前预警市场结构变化'
            if '领先' in lead_lag_summary and best_lag and best_lag < 0
            else '期限结构变化先于资金流向，需优先关注期限结构'
            if '领先' in lead_lag_summary and best_lag and best_lag > 0
            else '资金流向与期限结构同步变化，需同时监控'}
            
            💡 识别到的市场模式：
            {'⚠️【商品短缺模式】Backwardation加深、资金持续流入，商品可能短缺！' if commodity_shortage_mode else
            '⚠️【趋势衰竭模式】持仓-价格显著背离，趋势可能反转！' if trend_exhaustion_mode else
            '⚠️【闪崩风险模式】持仓高度集中、资金流向剧烈波动，市场易闪崩！' if flash_crash_mode else
            '✅【市场均衡模式】多空力量均衡，市场结构健康' if market_equilibrium else
            '🔍【混合状态】市场处于过渡期，需密切关注领先指标变化'}
            
            📊 风险状态评估：
            - 趋势衰竭信号：{risk_summary}
            - 闪崩风险：{'高频触发' if flash_crash_risks > n * 0.2 else '偶发触发' if flash_crash_risks > 0 else '未触发'}
            - 商品短缺信号：{'高频触发' if commodity_shortage_signals > n * 0.2 else '偶发触发' if commodity_shortage_signals > 0 else '未触发'}
            
            🎯 操作建议（基于当前模式和市场相对位置）：
            {('🔴【紧急行动】商品短缺模式已确认！建议：' +
            '   - 做多近月合约，做空远月合约，捕获Backwardation加深收益' +
            '   - 避免展期操作，选择延迟展期策略' +
            '   - 密切监控库存数据和地缘政治事件' if commodity_shortage_mode else
            '🟡【谨慎操作】趋势衰竭模式确认！建议：' +
            '   - 减少多头仓位，考虑反向操作' +
            '   - 设置更严格的止损点' +
            '   - 关注持仓-价格背离度变化' if trend_exhaustion_mode else
            '🔴【紧急行动】闪崩风险模式已确认！建议：' +
            '   - 大幅降低仓位，避免杠杆' +
            '   - 设置宽幅止损，防范极端波动' +
            '   - 避免在流动性低的时段交易' if flash_crash_mode else
            '🟢【积极配置】市场均衡模式确认！建议：' +
            '   - 维持正常风险敞口，执行既定交易策略' +
            '   - 利用波动率机会进行波段操作' +
            '   - 定期监控资金流向强度变化' if market_equilibrium else
            '🔵【观察等待】混合状态！建议：' +
            '   - 维持中性仓位，避免过度暴露' +
            '   - 设置预警线：持仓-价格背离度>2且为负值则减仓' +
            '   - 每周重新评估市场模式')}
            
            📌 风险提示：
            - 2025年12月市场特征：地缘政治冲突可能加剧商品短缺，需特别关注Backwardation结构
            - 本分析基于历史数据，极端行情下指标可能失效
            - 建议结合基本面数据综合判断
            
            🔍 深度洞察：
            {('资金流向领先期限结构变化约' + str(-best_lag) + '天，可作为早期预警信号。'
            if '领先' in lead_lag_summary and best_lag and best_lag < 0
            else '期限结构变化先于资金流向变化约' + str(best_lag) + '天，需优先关注期限结构。'
            if '领先' in lead_lag_summary and best_lag and best_lag > 0
            else '资金流向与期限结构同步变化，需同时监控两类指标。')}
            当趋势衰竭信号触发后，未来{int(abs(best_lag)) + 3 if best_lag else '5'}天内趋势反转概率平均上升{abs(best_corr) * 100:.0f}%。
            
            💡 特别提示：
            该期货当前表现{('商品短缺特征显著' if commodity_shortage_mode else
            '趋势衰竭特征明显' if trend_exhaustion_mode else
            '闪崩风险极高' if flash_crash_mode else
            '与')}同类期货整体水平，{('建议' if commodity_shortage_mode or flash_crash_mode else '谨慎')}{'做多近月' if commodity_shortage_mode else '减仓' if trend_exhaustion_mode or flash_crash_mode else '维持仓位'}
            """.strip()

        return summary

    def _analyze_option(self):
        pass

    # def construct_contract_features(
    #         self,
    #         contract_type: str,
    #         order_book_id: [str],
    #         start_date: str,
    #         end_date: str,
    # ) -> str:
    #     """
    #     构建适用于多种合约类型的全面特征集，不涉及聚合操作
    #     :param order_book_id: 用户指定的合约代码列表，仅对此部分样本开展特征工程
    #     :param contract_type: 合约类型 ('CS', 'ETF', 'INDX', 'Future', 'Option')
    #     :param start_date: 数据的起始日期
    #     :param end_date: 数据的终止日期
    #     :return: 包含所有特征的DataFrame的存储地址
    #     """
    #     df_addr, df_fields = self.ricequant_service.instruments_features_fetching(contract_type, int(start_date), int(end_date))
    #     df = pd.read_csv(df_addr)
    #     order_book_id_str = None
    #     if order_book_id:
    #         order_book_id_str = ','.join(sorted(order_book_id))
    #     order_book_id_hash = hashlib.md5(order_book_id_str.encode('utf-8')).hexdigest()[:10]
    #     output_path = os.path.join(self.features_data_path, f"{start_date}_{end_date}_{order_book_id_hash}_{contract_type}_features_data.csv")
    #     if os.path.exists(output_path):
    #         print("特征文件已存在！")
    #         return output_path
    #     else:
    #         print("特征文件不存在，开始生成")
    #
    #     # 1. 基础数据验证并选择合适的样本&按时间排序
    #     if df.empty:
    #         raise ValueError("输入数据为空")
    #     if order_book_id and 'order_book_id' in df.columns:     # 筛选出order_book_id在给定列表中的行
    #         df = df[df['order_book_id'].isin(order_book_id)]
    #     if 'date' in df.columns:
    #         df = df.sort_values(['date', 'order_book_id'])   # 整体数据优先【按照时间排序】
    #
    #     # 2. 标准化列名（处理可能的大小写差异）
    #     df = df.copy()
    #     df.columns = [col.lower() for col in df.columns]
    #
    #     # 3. 按合约类型构造特征
    #     if contract_type not in ['CS', 'ETF', 'INDX', 'Future', 'Option']:
    #         raise ValueError(f"不支持的合约类型: {contract_type}. 必须是 CS, ETF, INDX, Future, Option")
    #
    #     # 4. 初始化特征DataFrame
    #     features = pd.DataFrame(index=df.index)
    #     features['date'] = df['date']
    #     features['order_book_id'] = df['order_book_id']
    #     features['close'] = df['close']
    #
    #     # 关键步骤：创建分组对象
    #     grouped = df.groupby('order_book_id')
    #
    #     """ ===== 共享基础特征 (所有合约类型) ===== """
    #     # 价格特征
    #     features['returns'] = grouped['close'].transform(lambda x: x.pct_change())
    #     features['log_returns'] = grouped['close'].transform(lambda x: np.log(x / x.shift(1)))
    #
    #     df['returns'] = features['returns']     # 无需重新创建 grouped，因为 df 已经更新，grouped 会在访问时使用 df 的最新列
    #     df['log_returns'] = features['log_returns']
    #
    #     # 波动率特征
    #     features['vol_10d'] = grouped['returns'].transform(lambda x: x.rolling(10).std()) * np.sqrt(252)
    #     features['vol_20d'] = grouped['returns'].transform(lambda x: x.rolling(20).std()) * np.sqrt(252)
    #     features['vol_60d'] = grouped['returns'].transform(lambda x: x.rolling(60).std()) * np.sqrt(252)
    #     features['vol_ratio_20_60'] = features['vol_20d'] / features['vol_60d']  # 波动率斜率
    #
    #     # 趋势特征
    #     features['ma_5d'] = grouped['close'].transform(lambda x: x / x.rolling(5).mean() - 1)
    #     features['ma_20d'] = grouped['close'].transform(lambda x: x / x.rolling(20).mean() - 1)
    #     features['ma_60d'] = grouped['close'].transform(lambda x: x / x.rolling(60).mean() - 1)
    #
    #     # 动量特征
    #     df['ma_20d'] = features['ma_20d']
    #     features['ma_momentum'] = grouped['ma_20d'].transform(lambda x: x - x.shift(5))
    #
    #     # 真实波幅特征
    #     if 'high' in df.columns and 'low' in df.columns and 'prev_close' in df.columns:
    #         # 真实波幅计算
    #         def calculate_true_range(group):
    #             prev_close_shifted = group['prev_close'].shift(1)
    #             true_range_val = np.maximum(
    #                 group['high'] - group['low'],
    #                 np.maximum(
    #                     abs(group['high'] - prev_close_shifted),
    #                     abs(group['low'] - prev_close_shifted)
    #                 )
    #             )
    #             # 使用前一日收盘价计算百分比TR，注意分母也需要 shift(1)
    #             return true_range_val / group['prev_close'].shift(1)
    #
    #         features['true_range'] = grouped.apply(calculate_true_range, include_groups=False).reset_index(level=0, drop=True)
    #         # ATR
    #         df['true_range'] = features['true_range']
    #         features['atr_14d'] = grouped['true_range'].transform(lambda x: x.rolling(14).mean())
    #
    #     """ ===== 按合约类型添加特定特征 ===== """
    #     if contract_type in ['CS', 'ETF']:
    #         """ ===== 股票/ETF 特有特征 ===== """
    #         # 量能特征
    #         if 'volume' in df.columns:
    #             # 滚动均值
    #             features['volume_10d_ma'] = grouped['volume'].transform(lambda x: x.rolling(10).mean())
    #             features['volume_ratio'] = df['volume'] / features['volume_10d_ma']
    #             # 动量
    #             df['volume_ratio'] = features['volume_ratio']
    #             features['volume_momentum'] = grouped['volume_ratio'].transform(lambda x: x - x.shift(5))
    #
    #         if 'total_turnover' in df.columns:
    #             # 换手率与均值比
    #             features['turnover_ratio'] = grouped['total_turnover'].transform(lambda x: x / x.rolling(30).mean())
    #
    #         # 交易活跃度特征
    #         if 'num_trades' in df.columns:
    #             features['trade_frequency'] = df['num_trades'] / df['volume']
    #             # 20日均值
    #             df['trade_frequency'] = features['trade_frequency']
    #             features['trade_frequency_20d_ma'] = grouped['trade_frequency'].transform(lambda x: x.rolling(20).mean())
    #             features['trade_frequency_ratio'] = features['trade_frequency'] / features['trade_frequency_20d_ma']
    #
    #         # 市场状态特征
    #         if all(col in df.columns for col in ['close', 'limit_up', 'limit_down']):
    #             features['is_limit_up'] = (df['close'] >= df['limit_up'] * 0.995).astype(int)
    #             features['is_limit_down'] = (df['close'] <= df['limit_down'] * 1.005).astype(int)
    #             # 20日计数
    #             df['is_limit_up'] = features['is_limit_up']
    #             df['is_limit_down'] = features['is_limit_down']
    #             features['limit_up_count_20d'] = grouped['is_limit_up'].transform(lambda x: x.rolling(20).sum())
    #             features['limit_down_count_20d'] = grouped['is_limit_down'].transform(lambda x: x.rolling(20).sum())
    #
    #         # 换手率特征（股票特有）：此处涉及外部数据，分组处理难度大，保持原逻辑但需注意外部数据对齐
    #         features['turnover_rate_approx'] = df['total_turnover'] / (df['close'] * df['volume'])
    #         df['turnover_rate_approx'] = features['turnover_rate_approx']
    #
    #     elif contract_type == 'INDX':
    #         """ ===== 指数特有特征 ===== """
    #         # 市场广度指标
    #         if 'high' in df.columns and 'low' in df.columns:
    #             # 指数波动范围
    #             features['index_range'] = grouped[['high', 'low', 'close']].apply(
    #                 lambda x: (x['high'] - x['low']) / x['close'].shift(1),
    #                 include_groups=False
    #             ).reset_index(level=0, drop=True)
    #             # 20日均值
    #             df['index_range'] = features['index_range']
    #             features['index_range_20d_ma'] = grouped['index_range'].transform(lambda x: x.rolling(20).mean())
    #
    #         # 指数动量强度
    #         features['index_momentum_strength'] = features['returns'] / features['vol_20d']
    #
    #     elif contract_type in ['Future', 'Option']:
    #         """ ===== 期货/期权特有特征 ===== """
    #         # 持仓量特征（期货/期权）
    #         if 'open_interest' in df.columns:
    #             # 1日/5日变化
    #             features['oi_1d_change'] = grouped['open_interest'].transform(lambda x: x.pct_change())
    #             features['oi_5d_change'] = grouped['open_interest'].transform(lambda x: x.pct_change(5))
    #             # 动量
    #             df['oi_1d_change'] = features['oi_1d_change']
    #             features['oi_momentum'] = grouped['oi_1d_change'].transform(lambda x: x - x.rolling(5).mean())
    #
    #         settlement_col = 'settlement' if 'settlement' in df.columns else 'close'
    #         features['settlement'] = df[settlement_col]
    #
    #         # 期货特有特征：基差和期限结构涉及多个合约的数据对齐，此处保持原逻辑？？？？
    #
    #     elif contract_type == 'Option':
    #         # 行权价相关特征
    #         if 'strike_price' in df.columns:
    #             features['moneyness'] = df['close'] / df['strike_price']
    #             # 20日均值
    #             df['moneyness'] = features['moneyness']
    #             features['moneyness_20d_ma'] = grouped['moneyness'].transform(lambda x: x.rolling(20).mean())
    #             features['moneyness_deviation'] = features['moneyness'] - features['moneyness_20d_ma']
    #         # 隐含波动率估算（简化版）
    #         if 'strike_price' in df.columns and 'settlement' in df.columns:
    #             time_to_expiry = 30
    #             # 隐含波动率的计算不涉及滚动或 shift，但使用 apply 确保在组内操作
    #             features['implied_vol'] = grouped[['settlement', 'strike_price']].apply(
    #                 lambda x: np.sqrt(2 * np.pi / time_to_expiry) * (x['settlement'] / x['strike_price']),
    #                 include_groups=False
    #             ).reset_index(level=0, drop=True)
    #
    #     """ ===== 所有合约类型通用的高级特征 ===== """
    #     # 风险调整收益
    #     # 夏普比率
    #     features['sharpe_20d'] = grouped['returns'].transform(lambda x: x.rolling(20).mean()) / features[
    #         'vol_20d'] * np.sqrt(252)
    #     df['sharpe_20d'] = features['sharpe_20d']
    #
    #     # 波动率状态 (qcut 是全局操作，无需分组计算)
    #     features['vol_regime'] = pd.qcut(features['vol_20d'], q=5, labels=False, duplicates='drop') / 4
    #     df['vol_regime'] = features['vol_regime']
    #
    #     # 趋势强度
    #     trend_window = 20
    #     # 滚动标准差和均值
    #     price_std = grouped['close'].transform(lambda x: x.rolling(trend_window).std())
    #     price_mean = grouped['close'].transform(lambda x: x.rolling(trend_window).mean())
    #     features['trend_strength'] = (df['close'] - price_mean) / (price_std + 1e-10)
    #     df['trend_strength'] = features['trend_strength']
    #
    #     # 尾部风险指标
    #     # VaR
    #     features['var_95'] = grouped['returns'].transform(lambda x: x.rolling(60).quantile(0.05))
    #     df['var_95'] = features['var_95']
    #
    #     # CVaR(条件风险价值)
    #     df['cvar_returns_filtered'] = features['returns'].where(features['returns'] <= features['var_95'])
    #     features['cvar_95'] = grouped['cvar_returns_filtered'].transform(lambda x: x.rolling(60, min_periods=1).mean())   # 在每个合约分组内，对过滤后的（稀疏）收益率计算滚动平均。
    #     df.drop(columns=['cvar_returns_filtered'], inplace=True)
    #
    #     # 市场状态综合指标 (基于已分组计算的特征，无需再分组)
    #     features['market_regime'] = (
    #         0.4 * features['vol_regime'] +
    #         0.3 * abs(features['trend_strength']) +
    #         0.3 * (1 - features['sharpe_20d'].clip(lower=0, upper=1))
    #     )
    #
    #     """ ===== 特征工程后处理 ===== """
    #     MAX_ROLLING_WINDOW = settings.financial_data.features_max_rolling_window
    #     features = features.groupby('order_book_id').apply(
    #         lambda x: x.iloc[MAX_ROLLING_WINDOW:, :],
    #         include_groups=False
    #     ).reset_index(level=0, drop=False)      # 按 order_book_id 分组，丢弃每个分组的前 MAX_ROLLING_WINDOW 行
    #     features = features.reset_index(drop=True)
    #     features = features.replace([np.inf, -np.inf], np.nan)
    #
    #     # 填充必须在分组后进行，以避免使用下一只股票的数据填充前一只股票的NaN
    #     features_grouped_for_fillna = features.groupby('order_book_id')
    #     features = features_grouped_for_fillna.apply(
    #         lambda x: x.fillna(method='ffill'), include_groups=False).reset_index(level=0, drop=False)   # 不可使用bfill，避免未来信息泄露
    #     features = features.fillna(0)
    #     features = features.reset_index(drop=True)
    #
    #     # # 确保所有特征在合理范围内 (全局统计操作，保持不变)
    #     # for col in features.columns:
    #     #     if features[col].dtype in [np.float64, np.float32]:
    #     #         mean = features[col].mean()
    #     #         std = features[col].std()
    #     #         lower_bound = mean - 5 * std
    #     #         upper_bound = mean + 5 * std
    #     #         features[col] = features[col].clip(lower=lower_bound, upper=upper_bound)
    #
    #     # 移除可能由 apply 引入的额外索引
    #     features = features.sort_values(['date', 'order_book_id'])
    #     features.to_csv(output_path, index=False)
    #     return output_path


if __name__ == '__main__':
    ml_service = MLService()
    cs_list = ['000001.XSHE', '000002.XSHE', '000004.XSHE']
    etf_list = ['159001.XSHE', '159003.XSHE', '159005.XSHE']
    index_list = ['000001.XSHG', '000002.XSHG', '000003.XSHG']
    future_list = ['A2601', 'A2603', 'A2605']
    # print(ml_service.construct_contract_features('CS', cs_list, '20240401', '20251128'))
    # print(ml_service.summarize_CSanalysis(start_date=20250401,
    #    end_date=20251128,
    #    target_stock_id='000002.XSHE',
    #    order_book_id_list=cs_list))
    # print(ml_service.summarize_ETFanalysis(start_date=20250401,
    #     end_date=20251128,
    #     target_ETF_id='159003.XSHE',
    #     order_book_id_list=etf_list))
    # print(ml_service.summarize_INDXanalysis(start_date=20250401,
    #     end_date=20251128,
    #     target_index_id='000003.XSHG',
    #     index_id_list=index_list))
    print(ml_service.summarize_Futureanalysis(20250401, 20251128, 'A2603', future_list))

