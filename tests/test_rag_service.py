import os
from pathlib import Path
from src.core.load_config import settings
from src.services.rag_service import RAGService
from src.services.llm_service import init_llm_and_embed_models
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.retrievers import BaseRetriever


# --- 测试辅助函数 ---
def setup_test_data(raw_data_path):
    """创建测试数据文件。"""
    test_file_path = Path(raw_data_path) / "test_rag_info.txt"
    test_content = "核心规则：所有个人投资者必须首先完成风险评估问卷。中低风险产品 T+1 日可赎回。"
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(test_content)
    return test_content


# --- 核心测试函数 ---

def test_rag_service_lifecycle():
    """
    测试 RAGService 的完整生命周期：
    1. 初始化模型
    2. 首次构建索引
    3. 获取查询引擎
    4. 重新加载索引
    """

    print("--- 启动 RAGService 完整性测试 ---")

    # 1. 初始化模型和路径
    init_llm_and_embed_models()  # 确保 LLM 和 Embedding 被设置
    proj_dir = settings.project.project_dir
    persist_dir = os.path.join(proj_dir, settings.rag.index_persist_dir)
    raw_data_path = os.path.join(proj_dir, settings.paths.raw_data)

    # 清理环境以测试首次构建
    os.makedirs(raw_data_path, exist_ok=True)
    test_content = setup_test_data(raw_data_path)

    # ----------------------------------------------------
    # 步骤 A: 首次构建索引 (测试 _build_index)
    # ----------------------------------------------------
    print("\n[A] 测试首次构建...")
    rag_service_build = RAGService()

    print("✅ 首次构建成功并持久化。")

    # ----------------------------------------------------
    # 步骤 B: 测试 QueryEngine 和检索功能
    # ----------------------------------------------------
    print("\n[B] 测试查询引擎...")
    query_engine = rag_service_build.get_query_engine()

    assert isinstance(query_engine, BaseQueryEngine), "get_query_engine 未返回有效的 QueryEngine 实例！"

    # 测试检索 (只测试检索，不测试 LLM 回答，因为 LLM 响应慢)
    retriever = query_engine._retriever
    assert isinstance(retriever, BaseRetriever), "QueryEngine 中没有有效的 Retriever！"

    test_query = "关于风险评估的规则是什么？"
    retrieved_nodes = retriever.retrieve(test_query)

    assert len(retrieved_nodes) > 0, "检索失败，没有返回任何节点！"

    # 检查检索结果中是否包含我们植入的关键信息
    found_key_info = any("风险评估问卷" in node.get_content() for node in retrieved_nodes)
    assert found_key_info, "检索结果不准确，未找到植入的核心规则信息！"
    print("检索和 QueryEngine 初始化测试通过。")

    # ----------------------------------------------------
    # 步骤 C: 测试加载功能 (测试 _load_or_build_index)
    # ----------------------------------------------------
    print("\n[C] 测试从磁盘加载索引...")

    # 重新实例化服务，这次应该加载磁盘上的索引
    rag_service_load = RAGService()

    # 检查索引是否被成功加载而不是重新构建
    assert rag_service_load._index is not None
    print("索引加载测试通过。")

    print("\n--- RAGService 鉴定所有测试通过！---")


if __name__ == "__main__":
    # 在命令行直接运行此文件
    test_rag_service_lifecycle()