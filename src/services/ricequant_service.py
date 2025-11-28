import os
import pandas as pd
from datetime import datetime
from src.core.load_config import settings
import rqdatac
import csv
import json
import hashlib

class RiceQuantService:
    def __init__(self):
        rqdatac.init(uri=settings.financial_data.rice_quant_uri)
        print("初始化 RiceQuantService 成功！")
    def _update_instruments(self, type, date=None, market='cn', max_freq=settings.financial_data.update_frequency):
        """
        用于下载/更新合约信息，若在有效期内则无需更新
        """
        # 创建缓存目录
        cache_dir = os.path.join(settings.project.project_dir, settings.paths.financial_data)
        os.makedirs(cache_dir, exist_ok=True)

        # 处理日期参数
        if date is None:
            current_date_str = datetime.now().strftime("%Y%m%d")
        else:
            if isinstance(date, datetime):
                current_date_str = date.strftime("%Y%m%d")
            else:
                current_date_str = ''.join(filter(str.isdigit, str(date)))[:8]

        # 处理类型参数
        type_str = str(type)

        # 构建要查找的文件后缀
        file_suffix = f"{type_str}_instru.csv"

        # 在缓存目录下搜索匹配的文件
        existing_file = None
        for filename in os.listdir(cache_dir):
            if filename.endswith(file_suffix):
                existing_file = filename
                break

        newFileName = f"{current_date_str}_{type_str}_instru.csv"
        output_file_path = os.path.join(cache_dir, newFileName)
        if existing_file is not None:   # 存在目标类型的文件
            file_path = os.path.join(cache_dir, existing_file)
            file_date = existing_file.split('_')[0]

            # 执行数据更新逻辑
            if (datetime.now() - datetime.strptime(file_date, "%Y%m%d")).days > max_freq:
                print(f"合约基本信息缓存超过{max_freq}天，需要更新")
                data = rqdatac.all_instruments(type=type, date=int(current_date_str), market=market)
                data.to_csv(output_file_path, index=False)
                os.remove(file_path)
                print("更新完成！")
                return output_file_path
            else:
                print(f"合约基本信息缓存已存在且未超过{max_freq}天，无需更新")
                return file_path
        else:   # 不存在目标类型的文件
            print("合约基本信息缓存不存在，需要下载")
            data = rqdatac.all_instruments(type=type, date=int(current_date_str), market=market)
            data.to_csv(output_file_path, index=False)
            print("下载完成！")
            return output_file_path

    def _get_instruments_list(self, type):
        """
        获取当前合约类型下所有可交易的order_book_id
        :param type:
        :return: order_book_id字符串列表
        """
        file_addr = self._update_instruments(type=type)
        order_book_ids = []
        try:
            with open(file_addr, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    order_book_ids.append(row['order_book_id'])
        except Exception as e:
            print(e)
        return order_book_ids

    def _update_price_data(self, instru_type, start_date=None, end_date=None, frequency='1d',
                      fields=None, adjust_type='pre', skip_suspended=False):
        """
        用于下载/更新价格明细数据，支持增量更新避免重复下载
        """
        def convert_to_date_ranges(date_list):
            """
            将日期列表转换为连续的时间范围
            :param date_list: 日期列表，格式为yyyymmdd字符串
            :return: 时间范围列表，每个元素为(start_date, end_date)元组
            """
            if not date_list:
                return []

            # 排序日期列表
            sorted_dates = sorted(date_list)

            # 转换为datetime对象以便处理
            date_objects = [datetime.strptime(date, "%Y%m%d") for date in sorted_dates]

            ranges = []
            start_date = date_objects[0]
            end_date = date_objects[0]

            for i in range(1, len(date_objects)):
                # 如果当前日期与前一天连续（相差1天）
                if (date_objects[i] - end_date).days == 1:
                    end_date = date_objects[i]
                else:
                    # 不连续，保存当前范围并开始新的范围
                    ranges.append((start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")))
                    start_date = date_objects[i]
                    end_date = date_objects[i]

            # 添加最后一个范围
            ranges.append((start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")))
            return ranges

        # 创建缓存目录
        cache_dir = os.path.join(settings.project.project_dir, settings.paths.financial_data)
        os.makedirs(cache_dir, exist_ok=True)

        # 处理日期参数
        if start_date is None:
            start_date_str = datetime.now().strftime("%Y%m%d")
        else:
            if isinstance(start_date, datetime):
                start_date_str = start_date.strftime("%Y%m%d")
            else:
                start_date_str = ''.join(filter(str.isdigit, str(start_date)))[:8]

        if end_date is None:
            end_date_str = datetime.now().strftime("%Y%m%d")
        else:
            if isinstance(end_date, datetime):
                end_date_str = end_date.strftime("%Y%m%d")
            else:
                end_date_str = ''.join(filter(str.isdigit, str(end_date)))[:8]

        # 自动化获取order_book_ids列表
        order_book_ids = self._get_instruments_list(type=instru_type)

        # 生成文件名标识
        if fields is None:
            fields_str = "None"
        else:
            fields_str = ",".join(sorted(fields))  # 排序确保一致性
        fields_hash = hashlib.md5(fields_str.encode('utf-8')).hexdigest()[:10]  # 使用MD5生成稳定的哈希值，取前10位即可
        param_key = f"{instru_type}_{fields_hash}_{frequency}_{adjust_type}_{skip_suspended}"
        file_suffix = f"{param_key}_price.csv"

        # 查找现有文件
        existing_file = None
        existing_file_path = None
        for filename in os.listdir(cache_dir):
            if filename.endswith(file_suffix):
                existing_file = filename
                existing_file_path = os.path.join(cache_dir, filename)
                break

        # 生成新文件名
        new_filename = f"{start_date_str}_{end_date_str}_{file_suffix}"
        new_file_path = os.path.join(cache_dir, new_filename)

        # 如果没有现有文件，则直接下载全部数据
        if existing_file is None:
            print("价格数据不存在，开始下载...")
            data = rqdatac.get_price(order_book_ids, start_date=int(start_date_str), end_date=int(end_date_str),
                                    frequency=frequency, fields=fields, adjust_type=adjust_type,
                                    skip_suspended=skip_suspended, expect_df=True, market='cn')
            data.reset_index(inplace=True)
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date']).dt.strftime("%Y/%m/%d")
            if data is not None and not data.empty:
                data.to_csv(new_file_path, index=False)
                print("价格数据下载完成！")
                return new_file_path
            else:
                print("价格数据下载失败！")
                return None

        # 检查两个时间段是否存在交集
        existing_file_start_date, existing_file_end_date = existing_file.split('_')[0], existing_file.split('_')[1]

        existing_start_dt = datetime.strptime(existing_file_start_date, "%Y%m%d")
        existing_end_dt = datetime.strptime(existing_file_end_date, "%Y%m%d")
        existing_date_list = pd.date_range(start=existing_start_dt, end=existing_end_dt, freq='D').strftime("%Y%m%d").tolist()

        request_start_dt = datetime.strptime(start_date_str, "%Y%m%d")
        request_end_dt = datetime.strptime(end_date_str, "%Y%m%d")
        request_date_list = pd.date_range(start=request_start_dt, end=request_end_dt, freq='D').strftime("%Y%m%d").tolist()

        date_to_add = []   # 需要增量更新的日期，如果为空则需要全量下载
        if not (request_start_dt > existing_end_dt or request_end_dt < existing_start_dt):
            has_intersection = True   # 有交集的情况下date_to_add也可能为空（两个时间区间完全重合），因此需要通过标识符判断
            intersection_dates = list(set(existing_date_list) & set(request_date_list))
            date_to_add = list(set(request_date_list) - set(intersection_dates))
        date_to_add = convert_to_date_ranges(date_to_add)

        # 数据更新/重新下载
        if len(date_to_add) > 0:   # 有需要增量更新的日期
            print(f"执行增量更新")
            # 下载新增数据
            new_data = pd.DataFrame()
            for (tmp_start_date, tmp_end_date) in date_to_add:
                tmp_new_data = rqdatac.get_price(order_book_ids, start_date=int(tmp_start_date),
                                       end_date=int(tmp_end_date), frequency=frequency,
                                       fields=fields, adjust_type=adjust_type,
                                       skip_suspended=skip_suspended, expect_df=True, market='cn')
                tmp_new_data.reset_index(inplace=True)   #重置索引，保证列是统一的
                new_data = pd.concat([new_data, tmp_new_data], axis=0, ignore_index=True)
            if 'date' in new_data.columns:
                new_data['date'] = pd.to_datetime(new_data['date']).dt.strftime("%Y/%m/%d")
            if new_data is not None and not new_data.empty:
                existing_data = pd.read_csv(existing_file_path)
                updated_data = pd.concat([existing_data, new_data], axis=0, ignore_index=True)
                updated_data = updated_data.drop_duplicates()   # 去除重复行（基于所有列）
                updated_data.to_csv(new_file_path, index=False)
                os.remove(existing_file_path)
                print("价格数据增量更新完成！")
                return new_file_path
            else:
                print("没有新的价格数据，无需更新！")
                return existing_file_path
        else:
            if has_intersection:   # 有交集且无需增量更新
                print('价格数据已经存在，无需重复下载！')
                return existing_file_path
            else:   # 无交集，需全量下载
                print(f"现有数据无法进行增量更新，直接进行全量下载")
                data = rqdatac.get_price(order_book_ids, start_date=int(start_date_str), end_date=int(end_date_str),
                                        frequency=frequency, fields=fields, adjust_type=adjust_type,
                                        skip_suspended=skip_suspended, expect_df=True, market='cn')
                data.reset_index(inplace=True)
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date']).dt.strftime("%Y/%m/%d")
                if data is not None and not data.empty:
                    data.to_csv(new_file_path, index=False)
                    print("价格数据下载完成！")
                    return new_file_path
                else:
                    print("价格数据下载失败！")
                    return None

    def query_stock_info(self, type, query_by_code=None, query_by_symbol=None):
        """
        查询合约的基本信息
        :param type:
        :param query_by_code:
        :param query_by_symbol:
        :return: json格式，包含合约的基本信息
        """
        if not query_by_code and not query_by_symbol:
            return json.dumps({}, ensure_ascii=False, indent=2)

        file_addr = self._update_instruments(type=type)
        with open(file_addr, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if query_by_code:
                    if row['order_book_id'].startswith(query_by_code):
                        return json.dumps(row, ensure_ascii=False, indent=2)
                else:
                    if row['symbol'] == query_by_symbol:
                        return json.dumps(row, ensure_ascii=False, indent=2)
        return json.dumps({}, ensure_ascii=False, indent=2)

    def instruments_features_fetching(self, type, start_date, end_date):
        """
        自动根据用户选择的合约类型、数据时间范围进行下载/增量更新最新数据
        :param type:
        :param start_date:
        :param end_date:
        :return:数据地址、可使用的列字段
        """
        shared_FIELDS_LIST = [
            'close', 'high', 'low', 'total_turnover', 'volume', 'prev_close'
        ]
        optional_FIELDS_LIST = {'CS':['num_trades'], 'ETF':['num_trades'], 'INDX':[],
                                'Future':['open_interest', 'settlement'],
                                'Option':['strike_price', 'contract_multiplier']}
        try:
            fields_list = shared_FIELDS_LIST.extend(optional_FIELDS_LIST[type])
        except:
            print("输入的type有误！")
            fields_list = shared_FIELDS_LIST

        data_addr = self._update_price_data(instru_type=type, start_date=start_date, end_date=end_date, fields=fields_list)
        return data_addr, fields_list

if __name__ == '__main__':
    tmp = RiceQuantService()
    # print(tmp.query_stock_info('INDX', query_by_code='000001'))
    # print(tmp._get_instruments_list('CS'))
    # print(tmp._update_price_data('CS'))
    # print(tmp._update_price_data(instru_type='CS', start_date=20251124, end_date=20251128))
    FIELDS_LIST = [
        'close', 'high', 'low', 'total_turnover', 'volume', 'prev_close'
    ]
    print(tmp._update_price_data(instru_type='CS', start_date=20251124, end_date=20251128, fields=FIELDS_LIST))