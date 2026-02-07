import os
from dataclasses import dataclass

@dataclass
class Settings:
    WORKSPACE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "workspace"))
    DBS_DIR: str = "dbs"
    SCHEMA_DIR: str = "schema"
    HISTORY_DIR: str = "history"
    SAR_DIR: str = "sar"

    # 安全：默认限制查询返回行数（前端表格够用）
    DEFAULT_QUERY_LIMIT: int = 200

settings = Settings()
