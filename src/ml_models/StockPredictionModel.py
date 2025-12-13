import numpy as np
import pandas as pd
from src.services.ricequant_service import RiceQuantService
from src.core.load_config import settings
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
import hashlib
import os
warnings.filterwarnings('ignore')
ricequant_service = RiceQuantService()

class StockPredictionModel:
    def __init__(self, start_date:int, end_date:int, target_stock_id: str, order_book_id_list: list = None):
        self.start_date = start_date
        self.end_date = end_date
        self.target_stock_id = target_stock_id
        self.order_book_id_list = order_book_id_list
        if self.target_stock_id not in self.order_book_id_list:
            self.order_book_id_list.append(self.target_stock_id)
        self.model = None
        self.model_performance = None
        self.factor_data_for_prediction = None
        self.factor_columns = None
        self.predicted_excess_return = None

    def _get_technical_factors(self, start_date:int, end_date:int, order_book_id_list: list = None):
        """
        计算10个核心技术因子

        参数:
        df: 包含股票行情数据的DataFrame，必须包含以下列:
            - order_book_id: 股票代码
            - date: 日期
            - open, close, high, low: 开盘价、收盘价、最高价、最低价
            - limit_up, limit_down: 涨停价、跌停价
            - total_turnover: 成交额
            - volume: 成交量
            - prev_close: 昨日收盘价

        返回:
        包含10个技术因子的DataFrame
        """
        cs_features_list = ['open', 'close', 'high', 'low', 'limit_up', 'limit_down', 'total_turnover', 'volume',
                            'prev_close']
        df = ricequant_service.instruments_data_fetching(type='CS', start_date=start_date, end_date=end_date,
                                                              features_list=cs_features_list,
                                                              order_book_id_list=order_book_id_list)
        # 确保数据按股票代码和日期排序
        df = df.sort_values(['order_book_id', 'date']).reset_index(drop=True)

        # 创建结果DataFrame
        result = df[['order_book_id', 'date']].copy()

        # 1. MOM_20: 20日动量因子
        result['MOM_20'] = df.groupby('order_book_id')['close'].transform(
            lambda x: x / x.shift(20) - 1
        )

        # 2. REV_5: 5日反转因子
        result['REV_5'] = df.groupby('order_book_id')['close'].transform(
            lambda x: -(x / x.shift(5) - 1)
        )

        # 7. TURNOVER_VOL_RATIO: 风险调整后的流动性
        # 计算换手率（成交量/20日平均成交量）
        turnover = df.groupby('order_book_id')['volume'].transform(
            lambda x: x / x.rolling(20).mean()
        )
        # 计算20日波动率
        daily_return = (df['close'] - df['prev_close']) / df['prev_close']
        vol_ret_20 = daily_return.groupby(df['order_book_id']).transform(
            lambda x: x.rolling(20).std()
        )
        result['TURNOVER_VOL_RATIO'] = turnover / vol_ret_20

        # 8. VOL_PRICE_CORR_20: 20日量价相关系数
        def rolling_corr(x, y, window):
            """计算滚动相关系数"""
            cov = x.rolling(window).cov(y)
            std_x = x.rolling(window).std()
            std_y = y.rolling(window).std()
            return cov / (std_x * std_y)

        result['VOL_PRICE_CORR_20'] = rolling_corr(
            daily_return, df['volume'], 20
        ).groupby(df['order_book_id']).transform(lambda x: x)

        # 9. VOLUME_CONFIRM: 量能确认强度
        result['VOLUME_CONFIRM'] = np.sign(result['VOL_PRICE_CORR_20']) * turnover

        # 10. BODY_RATIO: K线实体比例
        result['BODY_RATIO'] = (df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-8)

        # 日度收益率
        result['DAILY_RETURN'] = df.groupby('order_book_id').apply(
            lambda x: (x['close'] - x['prev_close']) / x['prev_close']
        ).reset_index(level=0, drop=True)

        # 处理极端值和NaN
        for col in result.columns[2:]:  # 跳过order_book_id和date
            # 替换无穷大为NaN
            result[col] = result[col].replace([np.inf, -np.inf], np.nan)
            # 用中位数填充NaN（实际应用中可能需要更复杂的处理）
            result[col] = result.groupby('order_book_id')[col].transform(
                lambda x: x.fillna(x.median())
            )

        return result

    def _get_fundamental_factors(self, start_date:int, end_date: int, order_book_id_list: list = None):
        fundamental_factors_data = ricequant_service.factors_data_fetching('CS', settings.factors.fundamental_fields, start_date, end_date, order_book_id_list)
        return fundamental_factors_data

    def _build_dataset(self):
        """
        构造因子和超额收益率
        :return:
        """
        # 计算因子数据
        fundamental_factors = self._get_fundamental_factors(self.start_date, self.end_date, self.order_book_id_list)
        technical_factors = self._get_technical_factors(self.start_date, self.end_date, self.order_book_id_list)
        factors = pd.merge(fundamental_factors, technical_factors, on=['order_book_id', 'date'], how='inner')
        factors = factors.sort_values(['order_book_id', 'date']).reset_index(drop=True)

        # 计算无风险收益率
        hs_300 = ricequant_service.instruments_data_fetching('INDX', ['close', 'prev_close'],
                                                             self.start_date, self.end_date, ['000300.XSHG'])
        hs_300 = hs_300.sort_values(['order_book_id', 'date']).reset_index(drop=True)
        hs_300['RISK_FREE_DAILY_RETURN'] = (hs_300['close'] - hs_300['prev_close']) / hs_300['prev_close']
        hs_300.drop(columns=['order_book_id', 'close', 'prev_close'], inplace=True)

        # 对利率进行修正
        factors = pd.merge(factors, hs_300, on='date', how='left')
        factors['EXCESS_DAILY_RETURN'] = factors['DAILY_RETURN'] - factors['RISK_FREE_DAILY_RETURN']

        # 加入股票行业分类
        indus_info_addr = ricequant_service.industry_data_fetching(order_book_ids=self.order_book_id_list)
        indus_info = pd.read_csv(indus_info_addr)[['order_book_id', 'first_industry_name']]
        indus_info.rename(columns={'first_industry_name': 'industry'}, inplace=True)
        factors = pd.merge(factors, indus_info, on='order_book_id', how='left')

        # 因子标准化
        factor_columns = set(factors.columns.tolist()) - set(['order_book_id', 'date', 'industry', 'DAILY_RETURN', 'RISK_FREE_DAILY_RETURN', 'EXCESS_DAILY_RETURN'])
        y_column = 'EXCESS_DAILY_RETURN'

        for factor in factor_columns:
            factors[factor] = factors.groupby(['date', 'industry'])[factor].transform(
                lambda x: (x - x.mean()) / x.std()
            )

        return factors, list(factor_columns), y_column

    def _train(self):
        """
        从包含因子和目标变量的DataFrame训练一个LightGBM模型
        """
        df, feature_columns, target_column = self._build_dataset()
        self.factor_data_for_prediction = df[df['order_book_id'] == self.target_stock_id]
        self.factor_columns = feature_columns
        df = df[df['order_book_id'] != self.target_stock_id]
        n_cv_splits = settings.mlModels.cv_fold
        lgb_params = settings.mlModels.parameters
        train_ratio = settings.mlModels.train_ratio
        val_ratio = settings.mlModels.val_ratio
        cv_metric = settings.mlModels.cv_metric
        date_column = 'date'
        stock_column = 'order_book_id'

        # 1. 数据预处理和排序
        print("Step 1: Preprocessing and sorting data...")
        df_sorted = df.sort_values([date_column, stock_column]).reset_index(drop=True)
        all_dates = sorted(df_sorted[date_column].unique())

        # 2. 按时间划分数据集（确定最终测试集，CV在训练+验证部分进行）
        print(f"Step 2: Splitting data by time. Total dates: {len(all_dates)}")
        train_end_idx = int(len(all_dates) * train_ratio)
        val_end_idx = train_end_idx + int(len(all_dates) * val_ratio)

        train_dates_full = all_dates[:train_end_idx]
        val_dates_full = all_dates[train_end_idx:val_end_idx]
        test_dates = all_dates[val_end_idx:]  # 保留作为最终测试集

        # 3. 时间序列交叉验证
        print(f"Step 3: Performing Time Series CV with {n_cv_splits} splits...")
        cv_results = []
        models = []
        best_iterations = []  # 记录每个CV折的最佳迭代轮数

        # 确保总CV日期列表是纯净的列表
        total_cv_dates_list = train_dates_full + val_dates_full

        if len(total_cv_dates_list) < n_cv_splits:
            raise ValueError(f"Not enough dates ({len(total_cv_dates_list)}) for {n_cv_splits} CV splits.")

        # 手动创建时间序列分割
        split_size = len(total_cv_dates_list) // n_cv_splits
        fold_start = 0

        for i in range(n_cv_splits):
            # 确定当前折叠的验证集
            fold_end = fold_start + split_size
            if i == n_cv_splits - 1:  # 最后一个折叠包含剩余的所有日期
                fold_end = len(total_cv_dates_list)

            val_cv_dates = total_cv_dates_list[fold_start:fold_end]

            # 训练集是当前验证集之前的所有日期
            train_cv_dates = [d for d in total_cv_dates_list if d < val_cv_dates[0]]

            print(f"  Fold {i + 1}/{n_cv_splits}")
            print(
                f"    Train dates: {train_cv_dates[0] if train_cv_dates else 'None'} to {train_cv_dates[-1] if train_cv_dates else 'None'} (Count: {len(train_cv_dates)})")
            print(f"    Val dates: {val_cv_dates[0]} to {val_cv_dates[-1]} (Count: {len(val_cv_dates)})")

            if len(train_cv_dates) == 0:
                print(f"    Skipping fold {i + 1} due to insufficient training data.")
                fold_start = fold_end
                continue

            # 获取数据
            train_cv_df = df_sorted[df_sorted[date_column].isin(train_cv_dates)]
            val_cv_df = df_sorted[df_sorted[date_column].isin(val_cv_dates)]

            if train_cv_df.empty or val_cv_df.empty:
                print(f"    Skipping fold {i + 1} due to empty train or val set.")
                fold_start = fold_end
                continue

            # 准备特征和目标变量
            # 确保 feature_columns 是列表，避免pandas索引错误
            if not isinstance(feature_columns, list):
                feature_columns = list(feature_columns)

            X_train_cv = train_cv_df[feature_columns]  # 使用列表进行索引
            y_train_cv = train_cv_df[target_column]
            X_val_cv = val_cv_df[feature_columns]
            y_val_cv = val_cv_df[target_column]

            # 创建LightGBM数据集
            lgb_train_cv = lgb.Dataset(X_train_cv, y_train_cv)
            lgb_val_cv = lgb.Dataset(X_val_cv, y_val_cv, reference=lgb_train_cv)

            # 训练模型
            model_cv = lgb.train(
                lgb_params,
                lgb_train_cv,
                valid_sets=[lgb_train_cv, lgb_val_cv]
            )
            models.append(model_cv)

            # 记录最佳迭代轮数
            best_iterations.append(model_cv.best_iteration)
            print(f"    Fold {i + 1} - Best Iteration: {model_cv.best_iteration}")

            # 预测和评估
            print('Predicting...')
            y_pred_cv = model_cv.predict(X_val_cv)

            # 计算评估指标
            print('Evaluating...')
            eval_result = {}

            # 向量化预测：一次性预测整个验证集
            # 确保 feature_columns 是列表
            if not isinstance(feature_columns, list):
                feature_columns = list(feature_columns)
            X_val_cv_features = val_cv_df[feature_columns]
            y_pred_cv_vectorized = model_cv.predict(X_val_cv_features.values)  # 使用.values确保输入是numpy数组

            # 将预测结果添加到DataFrame中，方便后续分组操作
            val_cv_df_eval = val_cv_df.copy()
            val_cv_df_eval['pred'] = y_pred_cv_vectorized

            # 1. IC (信息系数) - 使用向量化方法计算每日IC并取平均
            def calculate_daily_ic_vectorized(group_df):
                # 使用 numpy 的 corrcoef 计算皮尔逊相关系数
                # corrcoef 返回的是相关系数矩阵，[0, 1] 位置是x和y的相关系数
                corr_matrix = np.corrcoef(group_df[target_column], group_df['pred'])
                return corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0  # 处理可能的NaN情况

            ic_by_date = val_cv_df_eval.groupby(date_column).apply(calculate_daily_ic_vectorized, include_groups=False)
            eval_result['ic'] = ic_by_date.mean()

            # 2. Rank IC (秩相关系数) - 使用pandas内置的corr方法计算每日秩相关系数并取平均
            def calculate_daily_rank_ic_vectorized(group_df):
                # 使用pandas的corr方法计算spearman秩相关
                # 这比手动排序后计算pearson相关更高效
                return group_df[target_column].corr(group_df['pred'], method='spearman')

            rank_ic_by_date = val_cv_df_eval.groupby(date_column).apply(calculate_daily_rank_ic_vectorized,
                                                                        include_groups=False)
            eval_result['rank_ic'] = rank_ic_by_date.mean()

            # 3. MAE - 直接计算整个验证集的MAE
            eval_result['mae'] = mean_absolute_error(val_cv_df_eval[target_column], val_cv_df_eval['pred'])

            # 4. MSE - 直接计算整个验证集的MSE
            eval_result['mse'] = mean_squared_error(val_cv_df_eval[target_column], val_cv_df_eval['pred'])

            cv_results.append(eval_result)
            print(
                f"    Fold {i + 1} - IC: {eval_result['ic']:.4f}, Rank IC: {eval_result['rank_ic']:.4f}, MAE: {eval_result['mae']:.4f}")

            # 更新下一个折叠的起始位置
            fold_start = fold_end

        if not models:
            raise ValueError("No models were trained during CV. Check data splits.")

        print("Step 4: Selecting best model based on CV metric:", cv_metric)
        # 根据指定的CV指标选择最优模型
        if cv_metric in ['ic', 'rank_ic']:
            # 对于IC类指标，选择平均值最高的
            scores = [res[cv_metric] for res in cv_results]
            best_model_idx = np.argmax(scores)
        elif cv_metric in ['mae', 'mse']:
            # 对于误差类指标，选择平均值最低的
            scores = [res[cv_metric] for res in cv_results]
            best_model_idx = np.argmin(scores)
        else:
            raise ValueError(f"Unknown cv_metric: {cv_metric}. Choose from 'ic', 'rank_ic', 'mae', 'mse'.")

        best_score = scores[best_model_idx]
        best_iteration_for_retrain = best_iterations[best_model_idx]  # 获取最佳迭代轮数
        print(f"Best Model from CV (Fold {best_model_idx + 1}): {cv_metric.upper()} = {best_score:.4f}")

        print("Step 5: Retraining final model on full train+val data using best iteration...")
        # 使用全部训练+验证数据重新训练
        full_train_df = df_sorted[df_sorted[date_column].isin(train_dates_full + val_dates_full)]
        # 确保 feature_columns 是列表
        if not isinstance(feature_columns, list):
            feature_columns = list(feature_columns)
        X_full_train = full_train_df[feature_columns]
        y_full_train = full_train_df[target_column]

        lgb_full_train = lgb.Dataset(X_full_train, y_full_train)

        # 创建不含早停参数的参数字典
        final_params = lgb_params.copy()
        # 移除早停参数，因为我们已确定迭代轮数
        if 'early_stopping_rounds' in final_params:
            del final_params['early_stopping_rounds']

        # 重新训练，使用CV中找到的最佳迭代轮数
        final_model = lgb.train(
            final_params,  # 使用不含早停参数的字典
            lgb_full_train,
            num_boost_round=best_iteration_for_retrain  # 使用最佳迭代轮数
        )
        print(f"Final model retrained for {best_iteration_for_retrain} rounds on full data.")
        print("Model training (and CV selection) completed.")
        self.model = final_model

        # --- 新增：在验证集和测试集上评估最终模型 ---
        print("Step 6: Evaluating final model on Validation and Test sets...")

        # 1. 准备验证集和测试集数据
        val_df = df_sorted[df_sorted[date_column].isin(val_dates_full)]
        test_df = df_sorted[df_sorted[date_column].isin(test_dates)]

        # 确保 feature_columns 是列表
        if not isinstance(feature_columns, list):
            feature_columns = list(feature_columns)

        X_val = val_df[feature_columns]
        y_val = val_df[target_column]
        X_test = test_df[feature_columns]
        y_test = test_df[target_column]

        # 2. 对验证集和测试集进行预测
        y_val_pred = final_model.predict(X_val.values)
        y_test_pred = final_model.predict(X_test.values)

        # 3. 定义评估函数
        def evaluate_on_dataset(y_true, y_pred, df_for_groupby, target_col, date_col):
            """计算单个数据集上的5个常见指标"""
            results = {}

            # 1. IC (信息系数)
            df_eval = df_for_groupby.copy()
            df_eval['pred'] = y_pred
            ic_by_date = df_eval.groupby(date_col).apply(
                lambda x: np.corrcoef(x[target_col], x['pred'])[0, 1] if len(x) > 1 else 0.0,
                include_groups=False
            )
            results['IC'] = ic_by_date.mean()

            # 2. Rank IC (秩相关系数)
            rank_ic_by_date = df_eval.groupby(date_col).apply(
                lambda x: x[target_col].corr(x['pred'], method='spearman') if len(x) > 1 else 0.0,
                include_groups=False
            )
            results['Rank_IC'] = rank_ic_by_date.mean()

            # 3. MAE (平均绝对误差)
            results['MAE'] = mean_absolute_error(y_true, y_pred)

            # 4. MSE (均方误差)
            results['MSE'] = mean_squared_error(y_true, y_pred)

            # 5. R2 Score (决定系数)
            results['R2_Score'] = r2_score(y_true, y_pred)

            return results

        # 4. 计算验证集指标
        val_metrics = evaluate_on_dataset(y_val, y_val_pred, val_df, target_column, date_column)

        # 5. 计算测试集指标
        test_metrics = evaluate_on_dataset(y_test, y_test_pred, test_df, target_column, date_column)

        # 6. 构建结果数据框
        results_df = pd.DataFrame({
            'Metric': ['IC', 'Rank_IC', 'MAE', 'MSE', 'R2_Score'],
            'Validation': [val_metrics['IC'], val_metrics['Rank_IC'], val_metrics['MAE'], val_metrics['MSE'],
                           val_metrics['R2_Score']],
            'Test': [test_metrics['IC'], test_metrics['Rank_IC'], test_metrics['MAE'], test_metrics['MSE'],
                     test_metrics['R2_Score']]
        })

        print("Final Model Evaluation Results:")
        print(results_df)

        # 可选：将结果存储到对象属性中
        self.model_performance = results_df

        return final_model, results_df  # 返回模型和评估结果

    def _save_model(self, file_path: str):
        """
        保存模型实例到本地文件
        """
        joblib.dump(self, file_path)
        print(f"模型已保存至: {file_path}")

    def _make_prediction(self):
        """
        使用模型进行预测
        """
        if not self.predicted_excess_return:
            X_pred = self.factor_data_for_prediction[list(self.factor_columns)].mean().values.reshape(1, -1)   # 使用历次的时间序列均值作为预测因子
            predicted_excess_return = self.model.predict(X_pred)[0]
            self.predicted_excess_return = predicted_excess_return
            return f"目标股票{self.target_stock_id}的次日超额收益率预测值为{predicted_excess_return:.6f}"
        else:
            return f"目标股票{self.target_stock_id}的次日超额收益率预测值为{self.predicted_excess_return:.6f}"

    @classmethod
    def runWholeProcedure(cls, start_date:int, end_date:int, target_stock_id: str, order_book_id_list: list = None) -> 'StockPredictionModel':
        """
        从指定路径加载StockPredictionModel实例（自动训练）
        """
        final_answer = ''
        selected_contracts_id_hash = 'None'
        if order_book_id_list:
            order_book_id_list_str = ','.join(sorted(order_book_id_list))
            selected_contracts_id_hash = hashlib.md5(order_book_id_list_str.encode('utf-8')).hexdigest()[:10]
        pack_name = f"{start_date}_{end_date}_{target_stock_id}_{selected_contracts_id_hash}_MLmodel.joblib"
        pack_save_path = os.path.join(settings.project.project_dir, settings.paths.ml_packs, pack_name)
        if os.path.exists(pack_save_path):
            model_instance = joblib.load(pack_save_path)
            print('成功从历史记忆中加载出训练好的模型！')
            final_answer += f'当前模型的性能表现为：\n{model_instance.model_performance}\n'
            final_answer += model_instance._make_prediction()
        else:
            print(f'历史记忆中不存在相关模型，开始训练...')
            model_instance = StockPredictionModel(start_date, end_date, target_stock_id, order_book_id_list)
            model, performance = model_instance._train()
            print('模型训练完成！')
            model_instance._save_model(pack_save_path)
            final_answer += model_instance._make_prediction()

        return final_answer


if __name__ == "__main__":
    cs_list = [
        # 传媒
        '000156.XSHE', '000607.XSHE', '000665.XSHE', '000676.XSHE', '000681.XSHE',
        # 电力设备
        '000009.XSHE', '000049.XSHE', '000159.XSHE', '000400.XSHE', '000533.XSHE',
        # 电子
        '000020.XSHE', '000021.XSHE', '000045.XSHE', '000050.XSHE', '000062.XSHE',
        # 房地产
        '000002.XSHE', '000006.XSHE', '000011.XSHE', '000014.XSHE', '000029.XSHE',
        # 纺织服饰
        '000017.XSHE', '000026.XSHE', '000726.XSHE', '000850.XSHE', '000955.XSHE',
        # 非银金融
        '000166.XSHE', '000415.XSHE', '000532.XSHE', '000563.XSHE', '000567.XSHE',
        # 钢铁
        '000629.XSHE', '000655.XSHE', '000708.XSHE', '000709.XSHE', '000717.XSHE',
        # # 公用事业
        # '000027.XSHE', '000037.XSHE', '000155.XSHE', '000407.XSHE', '000507.XSHE',
        # # 国防军工
        # '000519.XSHE', '000547.XSHE', '000561.XSHE', '000576.XSHE', '000638.XSHE',
        # # 环保
        # '000035.XSHE', '000068.XSHE', '000544.XSHE', '000546.XSHE', '000551.XSHE',
        # # 机械设备
        # '000008.XSHE', '000039.XSHE', '000157.XSHE', '000410.XSHE', '000425.XSHE',
        # # 基础化工
        # '000420.XSHE', '000422.XSHE', '000510.XSHE', '000525.XSHE', '000545.XSHE',
        # 计算机
        '000004.XSHE', '000034.XSHE', '000066.XSHE', '000158.XSHE', '000409.XSHE',
        # 家用电器
        '000016.XSHE', '000333.XSHE', '000404.XSHE', '000521.XSHE', '000541.XSHE',
        # 建筑材料
        '000012.XSHE', '000055.XSHE', '000401.XSHE', '000619.XSHE', '000672.XSHE',
        # 建筑装饰
        '000010.XSHE', '000032.XSHE', '000065.XSHE', '000498.XSHE', '000628.XSHE',
        # 交通运输
        '000088.XSHE', '000089.XSHE', '000099.XSHE', '000429.XSHE', '000520.XSHE',
        # 煤炭
        '000552.XSHE', '000571.XSHE', '000723.XSHE', '000937.XSHE', '000983.XSHE',
        # # 美容护理
        # '000615.XSHE', '001206.XSHE', '001328.XSHE', '002094.XSHE', '002511.XSHE',
        # # 农林牧渔
        # '000019.XSHE', '000048.XSHE', '000505.XSHE', '000592.XSHE', '000663.XSHE',
        # # 汽车
        # '000030.XSHE', '000338.XSHE', '000550.XSHE', '000559.XSHE', '000570.XSHE',
        # # 轻工制造
        # '000488.XSHE', '000659.XSHE', '000812.XSHE', '000910.XSHE', '001211.XSHE',
        # # 商贸零售
        # '000007.XSHE', '000058.XSHE', '000061.XSHE', '000151.XSHE', '000417.XSHE',
        # # 社会服务
        # '000428.XSHE', '000430.XSHE', '000524.XSHE', '000526.XSHE', '000558.XSHE',
        # # 石油石化
        # '000059.XSHE', '000096.XSHE', '000301.XSHE', '000554.XSHE', '000637.XSHE',
        # # 食品饮料
        # '000529.XSHE', '000568.XSHE', '000596.XSHE', '000639.XSHE', '000716.XSHE',
        # # 通信
        # '000063.XSHE', '000070.XSHE', '000586.XSHE', '000839.XSHE', '000889.XSHE',
        # # 医药生物
        # '000028.XSHE', '000078.XSHE', '000153.XSHE', '000403.XSHE', '000411.XSHE',
        # # 银行
        # '000001.XSHE', '001227.XSHE', '002142.XSHE', '002807.XSHE', '002839.XSHE',
        # # 有色金属
        # '000060.XSHE', '000408.XSHE', '000426.XSHE', '000506.XSHE', '000603.XSHE',
        # # 综合
        # '000025.XSHE', '000421.XSHE', '000523.XSHE', '000632.XSHE', '000652.XSHE'
    ]
    rst = StockPredictionModel.runWholeProcedure(20250401, 20251128, '000049.XSHE', cs_list)
    print(rst)



