import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from typing import Tuple
import ast
from src.core.load_config import settings

def init_llm_and_embed_models() -> Tuple[HuggingFaceLLM, HuggingFaceEmbedding]:
    """
    此函数负责处理复杂的 PyTorch 模型加载逻辑、路径拼接，并将结果封装为LlamaIndex 可用的 LLM 和 Embedding 对象。

    Returns:
        (HuggingFaceLLM, HuggingFaceEmbedding): 初始化后的 LLM 和 Embedding 模型实例。
    """

    # --- 解析模型路径和配置 ---
    llm_conf = settings.llm
    embed_conf = settings.embedding
    base_dir = settings.project.project_dir
    llm_full_path = os.path.join(base_dir, llm_conf.model_dir)
    embed_full_path = os.path.join(base_dir, embed_conf.model_dir)


    # 解析 max_memory_map 字符串为字典
    # YAML 中存储的 "{0: '15GiB'}" 是字符串，需要先替换单引号并解析为 Python 字典
    max_memory_map_dict = None
    try:
        if llm_conf.max_memory_map and llm_conf.max_memory_map.strip():
            parsed = ast.literal_eval(llm_conf.max_memory_map)
            if isinstance(parsed, dict):
                max_memory_map_dict = {int(k): v for k, v in parsed.items()}
    except (ValueError, SyntaxError) as e:
        pass


    # --- 检查 CUDA 可用性并设置设备 ---
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16  # 推荐使用 bfloat16 节省显存和加速
    else:
        device = "cpu"
        dtype = torch.float32


    # --- 加载 LLM 模型和分词器 ---
    tokenizer = AutoTokenizer.from_pretrained(llm_full_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        llm_full_path,
        max_memory=max_memory_map_dict,
        dtype=dtype,
        device_map="auto",  # 自动分配显存到可用设备
        trust_remote_code=True  # Qwen 系列模型通常需要这个
    )
    llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        context_window=llm_conf.context_window,
        max_new_tokens=llm_conf.max_new_tokens,
        generate_kwargs={"do_sample": settings.llm.do_sample, "temperature": settings.llm.temperature, "top_k": settings.llm.topk},
    )


    # --- 加载 Embedding 模型 ---
    # Embedding 模型通常不需要复杂的 device_map，直接指定 device 即可
    embed_model = HuggingFaceEmbedding(
        model_name=embed_full_path,
        device=device,
        trust_remote_code=True
    )


    # --- 设置 LlamaIndex 全局 Settings ---
    Settings.llm = llm
    Settings.embed_model = embed_model

    return llm, embed_model

if __name__ == "__main__":
    print("开始初始化 LLM 和 Embedding 模型...")
    try:
        llm, embed_model = init_llm_and_embed_models()
        print("模型初始化成功！")
        print(f"LLM 类型: {type(llm)}")
        print(f"Embedding 模型路径: {embed_model.model_name}")
    except Exception as e:
        print(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()