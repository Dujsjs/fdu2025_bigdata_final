from src.core.load_config import settings
import rqdatac
import os

rqdatac.init(uri=settings.financial_data.rice_quant_uri)
# tmp_data = rqdatac.get_price(['000001.XSHE', '000002.XSHE', '000004.XSHE'], frequency='1d', start_date=20251126, end_date=20251128)
# tmp_data.to_csv(os.path.join(settings.project.project_dir, settings.paths.raw_data, 'test_data.csv'), index=True)

tmp_data = rqdatac.get_factor(['000001.XSHE', '000002.XSHE'], ['pe_ratio_lyr', 'pcf_ratio_total_lyr'], start_date=20250626, end_date=20251128, expect_df=True, market='cn')
tmp_data.to_csv(os.path.join(settings.project.project_dir, settings.paths.raw_data, 'test_factor_data.csv'), index=True)

