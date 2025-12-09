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
        对股票日线数据进行深度分析
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

    def construct_contract_features(
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
    cs_list = ['000001.XSHE', '000002.XSHE', '000004.XSHE']
    # print(ml_service.construct_contract_features('CS', cs_list, '20240401', '20251128'))
    print(ml_service.summarize_CSanalysis(start_date=20250401,
       end_date=20251128,
       target_stock_id='000002.XSHE',
       order_book_id_list=cs_list))
