from src.core.load_config import settings
import json
import os
from datetime import datetime
import re
MEM_addr = os.path.join(settings.project.project_dir, settings.paths.memories)   # 记忆文件存储地址


def is_valid_stock_code(code_str: str) -> bool:
    """
    检查一个字符串是否是有效的 A 股或常见数据平台格式的股票代码。

    有效格式包括：
    - 上海交易所 (上交所)：6XXXXX.XSHG 或 6XXXXX.SS (科创板 68XXXX)
    - 深圳交易所 (深交所)：0XXXXX.XSHE 或 3XXXXX.XSHE 或 0XXXXX.SZ 或 3XXXXX.SZ

    参数:
        code_str (str): 要检查的股票代码字符串，例如 '600000.XSHG'。

    返回:
        bool: 如果代码格式合法，则返回 True；否则返回 False。
    """
    # 1. 使用正则表达式匹配基本结构：6位数字 + 点号 + 字母后缀
    match = re.match(r'^(\d{6})\.([A-Z]{2,4})$', code_str)

    if not match:
        # 如果格式不符合 "6位数字.后缀" 的模式
        return False

    stock_id, suffix = match.groups()

    # 2. 定义合法的交易所代码前缀和后缀

    # 沪市 (上海) 的代码前缀：6开头 (主板/科创板)
    SH_PREFIXES = ('6')
    # 沪市 (上海) 的合法后缀
    SH_SUFFIXES = ('XSHG', 'SS')

    # 深市 (深圳) 的代码前缀：0开头 (主板) 或 3开头 (创业板)
    SZ_PREFIXES = ('0', '3')
    # 深市 (深圳) 的合法后缀
    SZ_SUFFIXES = ('XSHE', 'SZ')

    # 3. 进行交易所匹配和校验

    # 沪市校验：代码以 6 开头，后缀必须是 XSHG 或 SS
    if stock_id.startswith(SH_PREFIXES):
        if suffix in SH_SUFFIXES:
            return True
        else:
            # 6开头的代码不能使用深圳的后缀
            print(f"校验失败: {code_str}，沪市代码使用了非沪市后缀。")
            return False

    # 深市校验：代码以 0 或 3 开头，后缀必须是 XSHE 或 SZ
    elif stock_id.startswith(SZ_PREFIXES):
        if suffix in SZ_SUFFIXES:
            return True
        else:
            # 0/3开头的代码不能使用上海的后缀
            print(f"校验失败: {code_str}，深市代码使用了非深市后缀。")
            return False

    # 4. 如果前缀既不是 6, 0, 也不是 3，则视为无效 A 股代码
    # A股代码前缀还有 8（北交所）和 4（新三板），但这里只涵盖最常见的沪深主板/创业板/科创板
    print(f"校验失败: {code_str}，非标准的 A 股代码前缀。")
    return False

def validate_and_convert_date(date_str: str) -> str | None:
    """
    验证并转换日期字符串为YYYYMMDD格式，失败则返回None
    """
    try:
        # 尝试解析日期字符串
        parsed_date = datetime.strptime(date_str, '%Y%m%d')
        # 转换回字符串格式并返回
        return int(parsed_date.strftime('%Y%m%d'))
    except ValueError:
        # 如果解析失败，返回None
        return None

def save_params_to_json(data: dict, filename: str):
    """
    将参数保存为JSON文件
    """
    try:
        # 使用 'w' 模式打开文件进行写入，并指定 utf-8 编码
        with open(filename, 'w', encoding='utf-8') as f:
            # indent=4 使输出的 JSON 文件具有漂亮的格式，易于阅读
            json.dump(data, f, indent=4)
        print(f"参数已成功保存到: {filename}")
    except TypeError as e:
        print(f"错误：数据类型错误。无法序列化字典：{e}")
    except IOError as e:
        print(f"写入文件时发生 I/O 错误: {e}")

def load_params_from_json(filename: str) -> dict | None:
    """
    从JSON文件中加载参数
    """
    # 检查文件是否存在
    if not os.path.exists(filename):
        print(f"错误：文件不存在: {filename}")
        return None

    try:
        # 使用 'r' 模式打开文件进行读取，并指定 utf-8 编码
        with open(filename, 'r', encoding='utf-8') as f:
            # json.load() 将 JSON 数据解析为 Python 字典
            data = json.load(f)
            print(f"参数已成功从 {filename} 加载。")
            return data

    except json.JSONDecodeError:
        print(f"错误：文件 {filename} 格式不正确，无法解析为 JSON。")
        return None
    except IOError as e:
        print(f"读取文件时发生 I/O 错误: {e}")
        return None

def get_param_CSanalysis():
    """
    为CS分析获取参数
    :return: 参数字典
    """
    data_dict = {
        "start_date": None,
        "end_date": None,
        "target_stock_id": None,
        "order_book_id_list": None
    }
    has_mem = False
    need_refreshment = False
    CS_mem_addr = os.path.join(MEM_addr, 'CSanalysis_mem.json')

    if os.path.exists(CS_mem_addr):
        data_dict = load_params_from_json(CS_mem_addr)
        if not data_dict:
            need_refreshment = True
        else:
            has_mem = True

    if has_mem:
        print(f"关于股票分析，历史记忆中的分析参数为：{data_dict}")
        temp_judge = input('您是否需要指定新的参数(Y/N)？')
        if temp_judge == 'Y':
            need_refreshment = True
            data_dict = {
                "start_date": None,
                "end_date": None,
                "target_stock_id": None,
                "order_book_id_list": None
            }
        else:
            print("继续使用历史记忆中的分析参数！")
    else:
        print("未找到历史记忆中的分析参数，请指定分析参数！")
        need_refreshment = True

    if need_refreshment:
        # 循环获取参数直到所有参数都已填充
        while True:
            # 获取开始日期
            if data_dict["start_date"] is None:
                start_date = input("请输入开始日期 (格式: YYYYMMDD): ")
                if start_date.strip():
                    start = validate_and_convert_date(start_date)
                    if start:
                        data_dict["start_date"] = start
                    else:
                        print(f"错误：无效的开始日期: {start_date}，请稍后重新输入")

            # 获取结束日期
            if data_dict["end_date"] is None:
                end_date = input("请输入结束日期 (格式: YYYYMMDD): ")
                if end_date.strip():
                    end = validate_and_convert_date(end_date)
                    if end:
                        if data_dict["start_date"] and data_dict["start_date"] <= end:
                            data_dict["end_date"] = end
                        else:
                            print("错误：开始日期不能大于结束日期，请稍后重新输入")
                    else:
                        print(f"错误：无效的结束日期: {end_date}，请稍后重新输入")

            # 获取目标股票ID
            if data_dict["target_stock_id"] is None:
                target_stock_id = input("请输入待研究的目标股票ID: （形如000001.XSHE）")
                if is_valid_stock_code(target_stock_id.strip()):
                    data_dict["target_stock_id"] = target_stock_id.strip()
                else:
                    print(f"错误：无效的股票ID: {target_stock_id}，请稍后重新输入")

            # 获取订单簿ID列表
            pass_order_book_id = False   # 跳过order_book_id的选择
            add_order_book = input("是否添加用于建模的股票ID列表(Y/N)? 注意：若不添加则默认选择全体股票分析，速度较慢！")
            try:
                if add_order_book.upper() == 'Y':
                    order_book_ids_input = input("请输入用于建模的股票ID列表，用英文逗号分割: （形如000001.XSHE,000002.XSHE）")
                    if order_book_ids_input.strip():
                        # 将输入的逗号分隔字符串转换为列表，并去除每个ID的首尾空格
                        order_book_ids = []
                        for id in order_book_ids_input.split(','):
                            if is_valid_stock_code(id.strip()):
                                order_book_ids.append(id.strip())
                            else:
                                print(f"错误：无效的股票ID: {id.strip()}，请稍后重新输入")

                        # 检查列表是否有效
                        if order_book_ids:
                            data_dict["order_book_id_list"] = order_book_ids
                            print(f"已添加股票ID列表: {order_book_ids}")
                            pass_order_book_id = True
                        else:
                            print("未检测到有效的股票ID，请重新输入")
                elif add_order_book.upper() == 'N':
                    # 用户选择不添加特定股票，保留空列表表示使用全部股票
                    print("将使用全部股票进行分析！")
                    pass_order_book_id = True
                else:
                    print("未检测到有效的股票ID，请重新输入")
            except AttributeError:
                # 处理add_order_book为None的情况
                print("未检测到有效的股票ID，请重新输入")
            except Exception as e:
                # 处理其他未预期的异常
                print(f"处理股票ID列表时发生错误: {e}")
                print("未检测到有效的股票ID，请重新输入")

            # 检查是否所有必需参数都已获取
            all_filled = (
                    data_dict["start_date"] is not None and
                    data_dict["end_date"] is not None and
                    data_dict["target_stock_id"] is not None and
                    pass_order_book_id
            )

            if all_filled:
                print("所有参数已获取完毕！")
                print(f"最终参数: {data_dict}")
                # 保存参数到记忆文件
                save_params_to_json(data_dict, CS_mem_addr)
                break
            else:
                missing_params = []
                if data_dict["start_date"] is None:
                    missing_params.append("开始日期")
                if data_dict["end_date"] is None:
                    missing_params.append("结束日期")
                if data_dict["target_stock_id"] is None:
                    missing_params.append("待研究的目标股票ID")
                if not pass_order_book_id:
                    missing_params.append("用于建模的股票ID列表")
                print(f"以下参数尚未填写完成: {', '.join(missing_params)}。请继续填写...")

    return data_dict

if __name__ == '__main__':
    get_param_CSanalysis()




