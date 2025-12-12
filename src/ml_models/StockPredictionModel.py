import numpy as np
from src.services.ricequant_service import RiceQuantService
from src.core.load_config import settings
ricequant_service = RiceQuantService()

class StockPredictionModel:
    def __init__(self, start_date:int, end_date:int, order_book_id_list: list = None):
        self.start_date = start_date
        self.end_date = end_date
        self.order_book_id_list = order_book_id_list

    def _get_technical_factors(start_date:int, end_date:int, order_book_id_list: list = None):
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

        # 处理极端值和NaN
        for col in result.columns[2:]:  # 跳过order_book_id和date
            # 替换无穷大为NaN
            result[col] = result[col].replace([np.inf, -np.inf], np.nan)
            # 用中位数填充NaN（实际应用中可能需要更复杂的处理）
            result[col] = result.groupby('order_book_id')[col].transform(
                lambda x: x.fillna(x.median())
            )

        return result

    def _get_fundamental_factors(start_date:int, end_date: int, order_book_id_list: list = None):
        fundamental_factors_data = ricequant_service.factors_data_fetching('CS', settings.factors.fundamental_fields, start_date, end_date, order_book_id_list)
        return fundamental_factors_data




if __name__ == "__main__":
    cs_list = ['000001.XSHE', '000002.XSHE', '000004.XSHE']
    model = StockPredictionModel(20250401, 20251128, cs_list)


