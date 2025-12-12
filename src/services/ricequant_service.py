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

    def instruments_data_fetching(self, type, features_list, start_date, end_date, order_book_id_list=None):
        """
        自动根据用户选择的合约类型、数据时间范围进行下载/增量更新最新数据
        :param type: 合约类型
        :param features_list: 特征列表
        :param start_date: int起始日期，yyyymmdd
        :param end_date: int终止日期，yyyymmdd
        :param order_book_id_list: order_book_id列表
        :return: dataframe数据框
        """
        data_addr = self._update_price_data(instru_type=type, start_date=start_date, end_date=end_date, fields=features_list)
        data = pd.read_csv(data_addr)
        data['date'] = pd.to_datetime(data['date'])
        start_date = datetime.strptime(str(start_date), "%Y%m%d")
        end_date = datetime.strptime(str(end_date), "%Y%m%d")
        if order_book_id_list is not None:
            data = data[
                (data['order_book_id'].isin(order_book_id_list)) &
                (data['date'] >= start_date) &
                (data['date'] <= end_date)
                ]
        else:    # 不给order_book_id_list的情况下默认取全部数据
            data = data[
                (data['date'] >= start_date) &
                (data['date'] <= end_date)
                ]
        data.reset_index(inplace=True, drop=True)
        return data

    def _update_shibor_data(self, start_date, end_date, shibor_range=['ON', '1W', '2W']):
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

        shibor_range_str = ','.join(shibor_range)
        shibor_range_hash = hashlib.md5(shibor_range_str.encode('utf-8')).hexdigest()[:10]
        file_name = f'{start_date_str}_{end_date_str}_{shibor_range_hash}_shibor_data.csv'
        output_file_path = os.path.join(settings.project.project_dir, settings.paths.financial_data, file_name)
        if os.path.exists(output_file_path):
            print("shibor数据已经存在，无需重复下载！")
            return output_file_path
        print("正在下载shibor数据...")
        shibor_data = rqdatac.get_interbank_offered_rate(start_date=int(start_date_str), end_date=int(end_date_str), fields=shibor_range, source='Shibor')
        shibor_data.reset_index(drop=False, inplace=True)
        shibor_data.to_csv(output_file_path, index=False)
        print("shibor数据下载完成！")
        return output_file_path

    def merge_shibor_data(self, data_to_merge: pd.DataFrame, start_date, end_date, shibor_range=['ON', '1W', '2W'], transfer_days: int=1):
        """
        将shibor数据加入到数据框中
        :param data_to_merge:
        :param start_date:
        :param end_date:
        :param shibor_range:
        :param transfer_days: shibor利率默认是年化利率，需要将其转化为对应时间粒度的数据
        :return:
        """
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

        shibor_data_addr = self._update_shibor_data(start_date=int(start_date_str), end_date=int(end_date_str), shibor_range=shibor_range)
        shibor_data = pd.read_csv(shibor_data_addr)
        data_to_merge['date'] = pd.to_datetime(data_to_merge['date']).dt.strftime('%Y-%m-%d')
        shibor_data['date'] = pd.to_datetime(shibor_data['date']).dt.strftime('%Y-%m-%d')

        if 'date' in data_to_merge.columns:
            # 如果有date列，则按date列进行左外连接合并
            merged_data = pd.merge(data_to_merge, shibor_data, on='date', how='left')
            merged_data[shibor_range] = merged_data[shibor_range].apply(lambda x: (1 + x/100) ** (transfer_days/360) - 1)
            return merged_data
        else:
            print("输入数据中缺少date列，无法合并Shibor数据")
            return data_to_merge

    def _update_factors_data(self, instru_type, factor_names, start_date=None, end_date=None):
        """
        用于下载/更新多个因子明细数据，支持增量更新避免重复下载
        注意: factor_names 应为因子名称列表，例如 ['roe', 'pe_ratio']。
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
        cache_dir = os.path.join(settings.project.project_dir, settings.paths.factor_data)  # 假设settings中有专门存放因子数据的路径
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

        # 生成文件名标识 (使用 factor_names 列表)
        # 对因子列表进行排序后拼接再哈希，确保顺序无关性
        sorted_factors_str = ",".join(sorted(factor_names))
        factors_hash = hashlib.md5(sorted_factors_str.encode('utf-8')).hexdigest()[:10]
        param_key = f"{instru_type}_{factors_hash}"
        file_suffix = f"{param_key}_factors.csv"  # 文件名不再包含具体因子名，因为可能很多

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
            print(f"因子数据不存在，开始下载...")
            # 直接请求多个因子
            data = rqdatac.get_factor(order_book_ids, factor_names, start_date=start_date_str, end_date=end_date_str,
                                      expect_df=True, market='cn')
            if data is not None and not data.empty:
                data.reset_index(inplace=True)  # 重置索引，通常 factor 数据会有多级索引
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date']).dt.strftime("%Y/%m/%d")
                data.to_csv(new_file_path, index=False)
                print(f"因子数据下载完成！")
                return new_file_path
            else:
                print(f"因子数据下载失败！")
                return None

        # --- 增量更新逻辑 ---
        # 这里需要特别注意：我们不能简单地按日期范围进行增量更新，
        # 因为我们不知道现有文件里的数据具体覆盖了哪些日期。
        # 最稳妥的方式是：
        # 1. 加载现有文件
        # 2. 获取现有文件中的日期范围
        # 3. 计算需要补充的日期范围
        # 4. 下载补充日期的数据
        # 5. 合并、去重、保存

        # 解析现有文件的日期范围
        existing_file_start_date, existing_file_end_date = existing_file.split('_')[0], existing_file.split('_')[1]
        existing_start_dt = datetime.strptime(existing_file_start_date, "%Y%m%d")
        existing_end_dt = datetime.strptime(existing_file_end_date, "%Y%m%d")

        request_start_dt = datetime.strptime(start_date_str, "%Y%m%d")
        request_end_dt = datetime.strptime(end_date_str, "%Y%m%d")

        # 检查请求时间范围是否完全包含在现有文件的时间范围内
        if request_start_dt >= existing_start_dt and request_end_dt <= existing_end_dt:
            print(f'请求的因子数据日期范围已完全存在于现有文件中，无需更新！')
            return existing_file_path

        # 计算需要增量更新的日期范围
        date_to_add = []
        # 情况1: 请求的开始日期早于现有文件的开始日期
        if request_start_dt < existing_start_dt:
            date_to_add.extend(pd.date_range(start=request_start_dt,
                                             end=min(existing_start_dt, request_end_dt) - pd.Timedelta(
                                                 days=1)).strftime("%Y%m%d").tolist())
        # 情况2: 请求的结束日期晚于现有文件的结束日期
        if request_end_dt > existing_end_dt:
            date_to_add.extend(pd.date_range(start=max(existing_end_dt, request_start_dt) + pd.Timedelta(days=1),
                                             end=request_end_dt).strftime("%Y%m%d").tolist())

        date_to_add = convert_to_date_ranges(date_to_add)

        if len(date_to_add) > 0:  # 有需要增量更新的日期
            print(f"执行因子数据增量更新")
            # 下载新增数据
            new_data = pd.DataFrame()
            for (tmp_start_date, tmp_end_date) in date_to_add:
                tmp_new_data = rqdatac.get_factor(order_book_ids, factor_names, start_date=tmp_start_date,
                                                  end_date=tmp_end_date, expect_df=True, market='cn')
                if tmp_new_data is not None and not tmp_new_data.empty:
                    tmp_new_data.reset_index(inplace=True)  # 重置索引，保证列是统一的
                    new_data = pd.concat([new_data, tmp_new_data], axis=0, ignore_index=True)

            if new_data is not None and not new_data.empty:
                if 'date' in new_data.columns:
                    new_data['date'] = pd.to_datetime(new_data['date']).dt.strftime("%Y/%m/%d")
                # 读取现有数据
                existing_data = pd.read_csv(existing_file_path)
                # 合并新旧数据
                updated_data = pd.concat([existing_data, new_data], axis=0, ignore_index=True)
                # 去除重复行（基于所有列，这对于因子数据通常是合理的，因为同一天同一支股票的因子值应该是唯一的）
                updated_data = updated_data.drop_duplicates()
                # 保存到新文件
                updated_data.to_csv(new_file_path, index=False)
                # 删除旧文件
                os.remove(existing_file_path)
                print(f"因子数据增量更新完成！")
                return new_file_path
            else:
                print(f"因子数据没有下载到新的增量数据，可能是因为指定日期范围内无交易日或因子数据。返回现有文件。")
                return existing_file_path
        else:
            # 如果 date_to_add 为空，意味着请求范围完全在现有范围内，或者没有跨越边界的请求，
            # 但第一种情况已在前面判断并返回。
            # 这里处理一种边界情况：请求范围与现有范围部分重叠，但不需要向两端扩展。
            # 例如：现有 [20230105, 20230110]，请求 [20230108, 20230112]，此时 date_to_add 只包含 [20230111, 20230112]。
            # 如果 date_to_add 真的为空，说明请求范围 <= 现有范围，也应在前面就返回。
            # 因此，这个 else 分支理论上不会执行到，除非逻辑判断有误。
            # 为了健壮性，可以将其视为需要重新全量下载，或者再次确认是否真的不需要更新。
            # 这里我们假设如果走到这一步，可能是日期计算有细微误差，但实际仍需全量更新。
            print(
                f"现有数据无法精确进行增量更新，尝试进行全量下载因子 (request: [{start_date_str}, {end_date_str}], existing: [{existing_file_start_date}, {existing_file_end_date}])")
            data = rqdatac.get_factor(order_book_ids, factor_names, start_date=start_date_str, end_date=end_date_str,
                                      expect_df=True, market='cn')
            if data is not None and not data.empty:
                data.reset_index(inplace=True)
                if 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date']).dt.strftime("%Y/%m/%d")
                data.to_csv(new_file_path, index=False)
                os.remove(existing_file_path)  # 更新时删除旧文件
                print(f"因子数据全量下载完成！")
                return new_file_path
            else:
                print(f"因子数据全量下载失败！返回现有文件。")
                return existing_file_path

    def factors_data_fetching(self, type, factors_list, start_date, end_date, order_book_id_list=None):
        """
        自动根据用户选择的合约类型、数据时间范围进行下载/增量更新最新因子数据
        :param type: 合约类型
        :param factors_list: 因子列表
        :param start_date: int起始日期，yyyymmdd
        :param end_date: int终止日期，yyyymmdd
        :param order_book_id_list: order_book_id列表
        :return: dataframe数据框
        """
        data_addr = self._update_factors_data(instru_type=type, factor_names=factors_list, start_date=start_date, end_date=end_date)
        data = pd.read_csv(data_addr)
        data['date'] = pd.to_datetime(data['date'])
        start_date = datetime.strptime(str(start_date), "%Y%m%d")
        end_date = datetime.strptime(str(end_date), "%Y%m%d")
        if order_book_id_list is not None:
            data = data[
                (data['order_book_id'].isin(order_book_id_list)) &
                (data['date'] >= start_date) &
                (data['date'] <= end_date)
                ]
        else:    # 不给order_book_id_list的情况下默认取全部数据
            data = data[
                (data['date'] >= start_date) &
                (data['date'] <= end_date)
                ]
        data.reset_index(inplace=True, drop=True)
        return data

    def industry_data_fetching(self, order_book_ids, source='sws', level=1, market='cn',
                               max_age_days=settings.financial_data.update_frequency):
        """
        用于下载/更新行业信息，若在有效期内则无需更新
        """
        # 创建缓存目录
        cache_dir = os.path.join(settings.project.project_dir, settings.paths.financial_data)
        os.makedirs(cache_dir, exist_ok=True)

        # 处理日期参数（作为文件名的一部分）
        current_date_str = datetime.now().strftime("%Y%m%d")

        # 处理 order_book_ids 参数
        # 将列表转换为排序后的字符串，然后生成哈希，以确保顺序无关性和文件名不会过长
        sorted_ids_str = ",".join(sorted(order_book_ids))
        ids_hash = hashlib.md5(sorted_ids_str.encode('utf-8')).hexdigest()[:10]

        # 将其他参数也纳入文件名考虑，确保不同参数组合的数据分开缓存
        params_str = f"{source}_{level}_{market}"
        # 如果 params_str 可能包含特殊字符，也可以考虑哈希处理，这里假设它是安全的
        file_suffix = f"indus_{ids_hash}_{params_str}.csv"

        # 在缓存目录下搜索匹配的文件
        existing_file = None
        for filename in os.listdir(cache_dir):
            if filename.endswith(file_suffix):
                existing_file = filename
                break

        new_filename = f"{current_date_str}_{file_suffix}"
        output_file_path = os.path.join(cache_dir, new_filename)

        if existing_file is not None:  # 存在目标参数的行业数据文件
            file_path = os.path.join(cache_dir, existing_file)
            file_date_str = existing_file.split('_')[0]  # 提取文件名开头的日期部分

            try:
                file_date = datetime.strptime(file_date_str, "%Y%m%d")
            except ValueError:
                # 如果文件名格式不正确，无法解析日期，则强制重新下载
                print(f"无法解析现有缓存文件 '{existing_file}' 的日期，将重新下载。")
                file_date = datetime.min  # 设置为一个很早的日期，确保下面的判断会触发下载

            # 执行数据更新逻辑
            if (datetime.now() - file_date).days > max_age_days:
                print(f"行业数据缓存超过 {max_age_days} 天，需要更新。")
                indus_data = rqdatac.get_instrument_industry(order_book_ids, source=source, level=level, market=market)
                if indus_data is not None and not indus_data.empty:
                    indus_data.reset_index(inplace=True, drop=False)
                    indus_data.to_csv(output_file_path, index=False)
                    os.remove(file_path)  # 删除旧缓存文件
                    print("行业数据更新完成！")
                    return output_file_path
                else:
                    print("行业数据下载失败或为空，返回旧缓存文件。")
                    return file_path  # 如果下载失败，返回旧文件
            else:
                print(f"行业数据缓存已存在且未超过 {max_age_days} 天，无需更新。")
                return file_path
        else:  # 不存在目标参数的行业数据文件
            print("行业数据缓存不存在，开始下载...")
            indus_data = rqdatac.get_instrument_industry(order_book_ids, source=source, level=level, market=market)
            if indus_data is not None and not indus_data.empty:
                indus_data.reset_index(inplace=True, drop=False)
                indus_data.to_csv(output_file_path, index=False)
                print("行业数据下载完成！")
                return output_file_path
            else:
                print("行业数据下载失败或为空，无法创建缓存文件。")
                return None  # 下载失败，返回 None 或抛出异常

if __name__ == '__main__':
    tmp = RiceQuantService()
    # print(tmp.query_stock_info('INDX', query_by_code='000001'))
    # print(tmp._get_instruments_list('CS'))
    # print(tmp._update_price_data('CS'))
    # print(tmp._update_price_data(instru_type='CS', start_date=20251124, end_date=20251128))

    # FIELDS_LIST = [
    #     'close', 'high', 'low', 'total_turnover', 'volume', 'prev_close'
    # ]
    # print(tmp._update_price_data(instru_type='CS', start_date=20251124, end_date=20251128, fields=FIELDS_LIST))
    # print(tmp._update_shibor_data(start_date=20251124, end_date=20251128, shibor_range=['1W']))

    # print(tmp._update_price_data(instru_type='CS', start_date=20250601, end_date=20251128, fields=['open', 'close', 'high', 'low', 'limit_up', 'limit_down', 'total_turnover', 'volume', 'num_trades', 'prev_close']))
    # print(tmp._update_price_data(instru_type='ETF', start_date=20250601, end_date=20251128, fields=['open', 'close', 'high', 'low', 'total_turnover', 'volume', 'num_trades', 'prev_close', 'iopv']))
    # print(tmp._update_price_data(instru_type='INDX', start_date=20250601, end_date=20251128, fields=['open', 'close', 'high', 'low']))
    # print(tmp._update_price_data(instru_type='Future', start_date=20250601, end_date=20251128, fields=['open', 'close', 'high', 'low', 'settlement', 'prev_settlement', 'open_interest', 'volume', 'total_turnover']))
    # print(tmp._update_price_data(instru_type='Option', start_date=20250601, end_date=20251128, fields=['open', 'close', 'high', 'low', 'settlement', 'prev_settlement', 'open_interest', 'volume', 'strike_price', 'contract_multiplier']))

    # print(tmp.instruments_data_fetching(type='CS', start_date=20251001, end_date=20251128, features_list=['open', 'close', 'high', 'low', 'limit_up', 'limit_down', 'total_turnover', 'volume', 'num_trades', 'prev_close'], order_book_id_list=['000001.XSHE', '000002.XSHE']))
    # tmp_data = tmp.factors_data_fetching('CS', settings.factors.fundamental_fields, 20250401, 20251128, ['000001.XSHE', '000002.XSHE', '000004.XSHE'])
    # print(tmp_data)

    file_path = os.path.join(settings.project.project_dir, settings.paths.financial_data, '20251208_CS_instru.csv')
    data = pd.read_csv(file_path)
    order_book_ids = data['order_book_id'].tolist()   # 全部的股票id
    print(tmp.industry_data_fetching(order_book_ids=order_book_ids))   # 获取行业数据

