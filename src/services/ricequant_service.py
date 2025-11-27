import os
from datetime import datetime
from src.core.load_config import settings
import rqdatac
import csv
import json

class RiceQuantService:
    def __init__(self):
        rqdatac.init(uri=settings.financial_data.rice_quant_uri)
        print("初始化 RiceQuantService 成功")
    def update_instruments(self, type, date=None, market='cn', cache_dir=settings.paths.financial_data, max_freq=settings.financial_data.update_frequency):
        """
        用于下载/更新合约信息，若在有效期内则无需更新
        """
        # 创建缓存目录
        cache_dir = os.path.join(settings.project.project_dir, cache_dir)
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
                print(f"缓存数据超过{max_freq}天，需要更新")
                data = rqdatac.all_instruments(type=type, date=int(current_date_str), market=market)
                data.to_csv(output_file_path, index=False)
                os.remove(file_path)
                print("更新完成！")
                return output_file_path
            else:
                print(f"数据已存在且未超过{max_freq}天，无需更新")
                return file_path
        else:   # 不存在目标类型的文件
            print("数据不存在，需要下载")
            data = rqdatac.all_instruments(type=type, date=int(current_date_str), market=market)
            data.to_csv(output_file_path, index=False)
            print("下载完成！")
            return output_file_path

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

        file_addr = self.update_instruments(type=type)
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


if __name__ == '__main__':
    # update_instruments(type='Option')
    tmp = RiceQuantService()
    print(tmp.query_stock_info('INDX', query_by_code='000001'))