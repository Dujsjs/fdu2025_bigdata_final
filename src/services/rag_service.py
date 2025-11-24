import os
from pathlib import Path
from typing import Optional

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
    SimpleDirectoryReader
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.readers.file import PDFReader
from src.core.load_config import settings
from src.services.llm_service import init_llm_and_embed_models


class RAGService:
    """
    RAGService 负责知识库的管理，包括索引的构建、加载和检索查询引擎的创建。
    """

    def __init__(self):
        # 项目绝对地址
        self.project_dir = settings.project.project_dir
        # 索引持久化目录
        self.persist_dir = os.path.join(self.project_dir, settings.rag.index_persist_dir)
        # 数据源目录
        self.raw_data_path = os.path.join(self.project_dir, settings.paths.raw_data)

        # 确保 LLM 和 Embedding 模型已初始化并设置到 Settings
        if Settings.llm is None or Settings.embed_model is None:
            init_llm_and_embed_models()

        # 确保持久化目录存在
        os.makedirs(self.persist_dir, exist_ok=True)
        os.makedirs(self.raw_data_path, exist_ok=True)

        # 初始化索引（加载或新建）
        self._index = self._load_or_build_index()

        # 在初始化后自动检查并更新索引
        # self._auto_update_index()

    def _get_file_extractor_map(self):
        """
        返回文件类型到阅读器的映射表。
        目前支持 TXT 和 PDF。
        """
        file_extractor_map = {
            ".pdf": PDFReader(),
            # 未来可在此处添加：".docx": DocxReader(),
            # 未来可在此处添加：".xlsx": ExcelReader(),
        }
        return file_extractor_map

    def _load_or_build_index(self):
        """尝试从磁盘加载索引，如果失败则构建新索引。"""
        persist_path = Path(self.persist_dir)

        # 检查是否存在持久化目录且包含至少一个关键文件
        if persist_path.exists() and any(persist_path.iterdir()):
            try:
                # 正确方式：使用 from_defaults(persist_dir=...)
                storage_context = StorageContext.from_defaults(persist_dir=str(persist_path))
                index = load_index_from_storage(storage_context)
                print("成功从磁盘加载现有索引")
                return index

            except Exception as e:
                print(f"加载索引失败: {e}，将重新构建...")
                return self._build_index()

        else:
            print("未找到持久化数据，开始构建新索引...")
            return self._build_index()

    def _build_index(self) -> Optional[VectorStoreIndex]:
        """构建全新索引（首次构建），不依赖任何现有数据。"""
        print("正在构建全新索引...")

        # 创建全新的存储上下文
        storage_context = StorageContext.from_defaults(
            vector_store=SimpleVectorStore(),
            docstore=SimpleDocumentStore(),
            index_store=SimpleIndexStore(),
        )

        # 读取数据
        reader = SimpleDirectoryReader(
            input_dir=self.raw_data_path,
            file_extractor=self._get_file_extractor_map(),
            recursive=True
        )
        documents = reader.load_data(show_progress=True, num_workers=4)

        if not documents:
            print("警告: 目录中没有找到可加载的文档，索引为空。")
            return None

        print(f"成功加载 {len(documents)} 份文档。")
        print(f"RAG 参数: Chunk Size={settings.rag.chunk_size}, Overlap={settings.rag.chunk_overlap}")

        # 定义 Ingestion Pipeline
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=settings.rag.chunk_size,
                    chunk_overlap=settings.rag.chunk_overlap
                ),
                Settings.embed_model
            ]
        )

        # 执行流水线并生成 Nodes
        nodes = pipeline.run(documents=documents)
        print(f"文档被分割成 {len(nodes)} 个节点 (Nodes)。")

        # 创建新索引
        print("正在创建向量索引...")
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
        )

        # 持久化
        print(f"索引持久化至: {self.persist_dir}")
        index.storage_context.persist(persist_dir=self.persist_dir)
        print("索引构建完成并持久化成功！")

        return index

    # def _auto_update_index(self) -> bool:
    #     """
    #     自动检查并更新索引（在初始化时自动调用）
    #     1. 检查是否有新/修改的文档
    #     2. 如果有，自动更新索引
    #     3. 无变化则不执行任何操作
    #
    #     Returns:
    #         bool: 是否执行了更新操作
    #     """
    #     print("\n开始自动检查索引更新...")
    #
    #     # 1. 检查是否有持久化索引
    #     if not self._index_exists():
    #         print("无法更新：没有找到现有索引，请先构建索引")
    #         return False
    #
    #     # 2. 加载现有存储上下文
    #     storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
    #
    #     # 3. 读取数据，只获取新增/修改的文档
    #     reader = SimpleDirectoryReader(
    #         input_dir=self.raw_data_path,
    #         file_extractor=self._get_file_extractor_map(),
    #         recursive=True
    #     )
    #     documents = reader.load_data(show_progress=True, num_workers=4)
    #
    #     if not documents:
    #         print("知识库无需更新！所有文档已是最新。")
    #         return False
    #
    #     print(f"检测到 {len(documents)} 份需要更新的文档。")
    #     print(f"RAG 参数: Chunk Size={settings.rag.chunk_size}, Overlap={settings.rag.chunk_overlap}")
    #
    #     # 4. 定义 Ingestion Pipeline
    #     pipeline = IngestionPipeline(
    #         transformations=[
    #             SentenceSplitter(
    #                 chunk_size=settings.rag.chunk_size,
    #                 chunk_overlap=settings.rag.chunk_overlap
    #             ),
    #             Settings.embed_model
    #         ]
    #     )
    #
    #     # 5. 执行流水线并生成 Nodes
    #     nodes = pipeline.run(documents=documents)
    #     print(f"新增文档被分割成 {len(nodes)} 个节点 (Nodes)。")
    #
    #     # 6. 加载现有索引并插入新节点
    #     print("正在更新向量索引...")
    #     index = load_index_from_storage(storage_context)
    #     index.insert_nodes(nodes)
    #
    #     # 7. 重新持久化
    #     print(f"持久化更新后的索引至: {self.persist_dir}")
    #     index.storage_context.persist(persist_dir=self.persist_dir)
    #     print("索引更新完成并持久化成功！")
    #
    #     # 8. 更新内部索引引用
    #     self._index = index
    #     return True
    #
    # def _index_exists(self) -> bool:
    #     """检查是否已有持久化索引存在"""
    #     persist_path = Path(self.persist_dir)
    #     required_files = [
    #         "docstore.json",
    #         "index_store.json",
    #         "default__vector_store.json",
    #         "graph_store.json",
    #         "image__vector_store.json"
    #     ]
    #     return persist_path.exists() and all((persist_path / f).exists() for f in required_files)

    def get_query_engine(self) -> BaseQueryEngine:
        """
        向 Agent 层暴露的接口：获取一个用于查询的 QueryEngine。

        Returns:
            BaseQueryEngine: 可用于执行查询的引擎实例

        Raises:
            RuntimeError: 当索引未成功构建时
        """
        if self._index is None:
            raise RuntimeError("RAG 索引尚未成功构建，无法提供查询服务。")

        # QueryEngine：结合检索器和 LLM 进行问答
        query_engine = self._index.as_query_engine(
            llm=Settings.llm,
            similarity_top_k=settings.rag.similarity_top_k
        )
        return query_engine