# 注意：此模块暂时停用！
import os
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass
from src.ml_models.ContractAnalysisModel import ContractAnalysisModel
from src.services.ml_service import MLService
from src.core.load_config import settings
import hashlib
import joblib

@dataclass
class MLPackConfig:
    """
    MLPack的配置信息，由用户指定
    """
    start_date: str
    end_date: str
    contract_type: str
    selected_contracts_id: List[str]   # 指定用于训练模型的合约代码
    shibor_type: str
    predict_days: int

@dataclass
class MLPackAddresses:
    """
    MLPack中涉及的各种文件地址（绝对地址），MLPack初始化时自动生成
    """
    features_data_addr: str = ""
    contract_analysis_model_addr: str = ""

class MLPack:
    """
    MLPack将配置、模型等信息打包到一起，每个类型的合约对应一个实例，
    相当于总共最多有5个实例 (CS, ETF, INDX, Future, Option)
    """
    def __init__(self, config: MLPackConfig):
        """
        初始化MLPack实例（只存模型地址，不存实例，惰性加载）

        Args:
            config: MLPack配置信息
        """
        # 配置和数据路径信息
        self.config = config
        self.addr = MLPackAddresses()
        self.X_sample_for_analysis = None

        print('MLPack 预加载完成')

    def _train_model(self):
        """
        训练模型并自动化保存模型到本地路径，返回训练好的模型
        """
        # 确定模型保存地址，判断本地是否存在已经训练好的模型
        selected_contracts_id_str = ','.join(sorted(self.config.selected_contracts_id))
        selected_contracts_id_hash = hashlib.md5(selected_contracts_id_str.encode('utf-8')).hexdigest()[:10]
        model_name = f"{self.config.start_date}_{self.config.end_date}_{selected_contracts_id_hash}_{self.config.contract_type}_{self.config.shibor_type}_{self.config.predict_days}_model.joblib"
        model_save_path = os.path.join(settings.project.project_dir, settings.paths.trained_models, model_name)

        if os.path.exists(model_save_path):   # 说明可能是实际模型没有与当前实例链接上，尝试建立链接即可
            print('已经存在训练好的模型，无需再次训练！')
            self.addr.contract_analysis_model_addr = model_save_path
            return self._load_model()

        print('开始训练模型...')
        # 构建模型实例
        ml_service = MLService()
        trained_model = ContractAnalysisModel(self.config.contract_type)

        # 特征数据获取
        feature_data_path = ml_service.construct_contract_features(self.config.contract_type, self.config.selected_contracts_id, self.config.start_date, self.config.end_date)
        self.addr.features_data_addr = feature_data_path
        origin_feature_data = pd.read_csv(feature_data_path)

        # 模型训练
        X_train, y_train = trained_model.preprocess_features_data(origin_feature_data, self.config.start_date, self.config.end_date, self.config.shibor_type, self.config.predict_days)
        trained_model.train(X_train, y_train)

        # 模型保存
        self.X_sample_for_analysis = X_train.iloc[-1].copy()    # 取最后一行数据作为预测数据
        self.addr.contract_analysis_model_addr = model_save_path
        trained_model.save_model(model_save_path)
        return trained_model

    def _load_model(self):
        """
        加载已训练的模型
        """
        if not self.addr.contract_analysis_model_addr:    # 本地无模型或模型未与当前实例连接上
            trained_model = self._train_model()
            self._save_pack(force_overwrite=True)
            return trained_model
        try:
            trained_model = ContractAnalysisModel.load_model(self.addr.contract_analysis_model_addr)
        except:
            trained_model = self._train_model()
            self._save_pack(force_overwrite=True)
        return trained_model

    def _save_pack(self, force_overwrite: bool = False) -> None:
        """
        保存整个pack
        """
        selected_contracts_id_str = ','.join(sorted(self.config.selected_contracts_id))
        selected_contracts_id_hash = hashlib.md5(selected_contracts_id_str.encode('utf-8')).hexdigest()[:10]
        pack_name = f"{self.config.start_date}_{self.config.end_date}_{selected_contracts_id_hash}_{self.config.contract_type}_{self.config.shibor_type}_{self.config.predict_days}_pack.joblib"
        pack_save_path = os.path.join(settings.project.project_dir, settings.paths.ml_packs, pack_name)
        if os.path.exists(pack_save_path) and not force_overwrite:
            print('已经存在该pack，无需再次保存！')
            return pack_save_path
        joblib.dump(self, pack_save_path)
        return pack_save_path

    def do_analysis(self) -> Dict[str, Any]:
        """
        执行完整分析
        """
        model = self._load_model()
        report = model.generate_investment_report(self.X_sample_for_analysis)
        return report

    @classmethod
    def load_or_build_pack(cls, config: MLPackConfig) -> 'MLPack':
        """
        从指定路径加载pack
        """
        selected_contracts_id_str = ','.join(sorted(config.selected_contracts_id))
        selected_contracts_id_hash = hashlib.md5(selected_contracts_id_str.encode('utf-8')).hexdigest()[:10]
        pack_name = f"{config.start_date}_{config.end_date}_{selected_contracts_id_hash}_{config.contract_type}_{config.shibor_type}_{config.predict_days}_pack.joblib"
        pack_save_path = os.path.join(settings.project.project_dir, settings.paths.ml_packs, pack_name)
        if os.path.exists(pack_save_path):
            pack_instance = joblib.load(pack_save_path)
            print('pack加载成功！')
            return pack_instance
        else:
            pack_instance = MLPack(config)
            pack_addr = pack_instance._save_pack()
            print(f'pack创建成功：{pack_addr}！')
            return pack_instance

if __name__ == '__main__':
    cs_list = [
        "000001.XSHE",
        "000002.XSHE",
        "000004.XSHE",
        "000006.XSHE",
        "000007.XSHE",
        "000008.XSHE",
        "000009.XSHE",
        "000010.XSHE",
        "000011.XSHE",
        "000012.XSHE",
        "000014.XSHE",
        "000016.XSHE",
        "000017.XSHE",
        "000019.XSHE",
        "000020.XSHE",
        "000021.XSHE",
        "000025.XSHE",
        "000026.XSHE",
        "000027.XSHE",
        "000028.XSHE",
        "000029.XSHE",
        "000030.XSHE",
        "000031.XSHE",
        "000032.XSHE",
        "000034.XSHE",
        "000035.XSHE",
        "000036.XSHE"
    ]
    config = MLPackConfig(
        contract_type='CS',
        selected_contracts_id=cs_list,
        start_date='20240401',
        end_date='20251128',
        shibor_type='1W',
        predict_days=3
    )
    pack = MLPack.load_or_build_pack(config)
    report = pack.do_analysis()
    print(report)
