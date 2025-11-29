import os
import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv

# --- 加载环境变量 ---
# 获取当前文件 (src/core/load_config.py) 的父级目录，向上推导找到 config/.env
BASE_DIR = Path(__file__).resolve().parent.parent.parent


# --- 定义配置的数据结构 (Pydantic Models) ---
class ProjectInfo(BaseModel):
    name: str
    author: str
    project_dir: str

class LLMConfig(BaseModel):
    model_name: str
    model_dir: str
    cache_dir: str
    context_window: int
    max_new_tokens: int
    max_memory_map: str
    temperature: float
    topk: int
    do_sample: bool

class EmbeddingConfig(BaseModel):
    model_name: str
    model_dir: str
    cache_dir: str

class RAGConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int
    similarity_top_k: int
    index_persist_dir: str

class financialDataConfig(BaseModel):
    update_frequency: int
    rice_quant_uri: str
    features: dict

class mlModelConfig(BaseModel):
    va_para: dict
    cv_fold: int

class PathsConfig(BaseModel):
    raw_data: str
    financial_data: str
    processed_data: str
    ml_models: str

class AppConfig(BaseModel):
    project: ProjectInfo
    llm: LLMConfig
    embedding: EmbeddingConfig
    financial_data: financialDataConfig
    rag: RAGConfig
    mlModels: mlModelConfig
    paths: PathsConfig


# --- 配置加载函数 ---
def load_config() -> AppConfig:
    config_path = BASE_DIR / "config" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        yaml_content = yaml.safe_load(f)

    # 将字典转换为 Pydantic 对象，这里会自动验证类型
    # 如果 yaml 里把 chunk_size 写成了字符串，这里会报错提醒
    return AppConfig(**yaml_content)


# --- 初始化单例对象 ---（确保整个应用中只有一个settings对象实例，避免了每次需要配置时都重复读取和解析 YAML 文件，提高了效率并保证了全局配置的一致性）
try:
    settings = load_config()
    # print(f"Configuration loaded successfully.")
except Exception as e:
    print(f"Failed to load configuration: {e}")
    raise e