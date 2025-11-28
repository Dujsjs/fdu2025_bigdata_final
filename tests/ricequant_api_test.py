from src.core.load_config import settings
import rqdatac
import os

rqdatac.init(uri=settings.financial_data.rice_quant_uri)
tmp_data = rqdatac.get_price(['000001.XSHE', '000002.XSHE', '000004.XSHE'], frequency='1d', start_date=20251126, end_date=20251128)
tmp_data.to_csv(os.path.join(settings.project.project_dir, settings.paths.raw_data, 'test_data.csv'), index=True)