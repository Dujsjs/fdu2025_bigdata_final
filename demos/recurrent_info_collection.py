# demo：循环提问收集用户个人信息，直至全部收集到
import json

# 初始用户输入，在实际循环中会动态更新
user_input = ""
user_info = {"name": None, "age": None, "address": None}
required_fields = list(user_info.keys())  # ['name', 'age', 'address']

print("--- 启动信息收集系统 ---")
print(f"当前所需字段: {required_fields}")

# 循环直到所有必填字段都已收集
while not all(user_info.get(field) for field in required_fields):

    # 动态生成提问
    # 找出第一个尚未填写的必填字段
    next_missing_field = next(
        (field for field in required_fields if not user_info.get(field)),
        None  # 如果没有缺失的, 返回 None
    )

    # 准备一个友好的问题
    question = ""
    if next_missing_field == "name":
        question = "首先，请问您的全名是什么？"
    elif next_missing_field == "address":
        question = "好的，接下来我需要您的详细地址。"
    elif next_missing_field == "age":
        question = "请问您的年龄是多少？"  # 补充一个 age 的提问，以覆盖所有字段
    else:
        # 如果所有必填项都已收集，理论上不会进入这里，但为保障
        break

    print(f"\n机器人: {question}")
    user_input = input("您: ")

    # --- 信息提取部分 ---
    try:
        extraction_prompt = f"""
        你是一个信息提取专家。从下面的用户输入中，提取姓名 (name)、年龄 (age) 和地址 (address)。
        严格按照以下 JSON 格式输出，不要添加任何解释或额外的文本。
        如果某项信息不存在，请将其值设为 null。
        
        用户输入: "{user_input}"
        
        JSON输出:
        """

        # 调用 LLM 进行信息提取
        response = llm.complete(extraction_prompt)   #此处要替换为真实的llm-api调用
        extracted_text = response.text.strip()

        # 清理 LLM 可能返回的 markdown 代码块标记
        if extracted_text.startswith("```json"):
            # 假设标记总是 ```json\n...\n``` 形式
            extracted_text = extracted_text[7:-3].strip()

            # 解析 LLM 返回的 JSON 字符串
        extracted_data = json.loads(extracted_text)

        # 更新 user_info 字典, 只更新有值的新信息
        for key, value in extracted_data.items():
            if value is not None and key in user_info:
                # 只有在当前字段为空时才更新, 防止覆盖已有信息
                if user_info[key] is None:
                    user_info[key] = value
                    print(f"[系统提示: 已记录 {key}: {value}]")

    except (json.JSONDecodeError, TypeError) as e:
        print(f"[系统提示: LLM返回数据解析错误或类型错误: {e}]")

    except Exception as e:
        print(f"[系统提示: 与模型通信时发生未知错误: {e}]")

print("\n--- 收集完成 ---")
print("机器人: 恭喜！所需信息已全部收集。")
print(f"最终信息: {user_info}")
print("-" * 30)