from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List, Optional



class HistoryStatsResponse(BaseModel):
    ok: bool = True
    db_id: str = ""
    totalQueries: int = 0
    todayQueries: int = 0
    successRate: Optional[float] = None
    avgLatencyMs: Optional[int] = None

class UploadResponse(BaseModel):
    db_id: str
    db_path: str

class ParseSchemaRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    db_id: str
    k_examples: int = Field(default=2, alias="k")


class SchemaOverview(BaseModel):
    db_id: str
    tables: int
    columns: int
    foreign_keys: int
    schema_text: Optional[str] = None
    schema_dict: Optional[Dict[str, Any]] = None


class EnrichSchemaRequest(BaseModel):
    db_id: str
    question: str
    evidence: str = ""
    k: int = 2


class EnrichSchemaResponse(BaseModel):
    db_id: str
    schema_text_enriched: str
    sample_text: str


class GenerateRequest(BaseModel):
    db_id: str
    question: str
    options: Dict[str, Any] = Field(default_factory=dict)


class CandidateScore(BaseModel):
    # ✅ populate_by_name: 允许传 schema 或 schema_info 都能解析
    # ✅ ser_json_by_alias: 输出时用 alias（保证前端还是拿到 schema 字段）
    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)

    executability: Dict[str, Any]
    schema_info: Dict[str, Any] = Field(..., alias="schema")  # 对外仍叫 schema
    examples: Dict[str, Any]
    final_score: float


class Candidate(BaseModel):
    id: str
    sql: str
    scores: CandidateScore


class GenerateResponse(BaseModel):
    request_id: str
    candidates: List[Candidate]
    selected_sql: str


class ExecuteRequest(BaseModel):
    db_id: str
    sql: str


class ExecuteResponse(BaseModel):
    ok: bool
    sql: Optional[str] = None
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    elapsed_ms: int
    error: Optional[str] = None

    charts_suggest: List[Any] = []   # ✅ 加上这一行


class FeedbackRequest(BaseModel):
    db_id: str
    question: str
    selected_sql: str
    ok: bool
    corrected_sql: Optional[str] = ""

    gen_ms: Optional[int] = 0
    exec_ms: Optional[int] = 0
    total_ms: Optional[int] = 0


class FeedbackResponse(BaseModel):
    ok: bool
    history_id: int


class HistoryListResponse(BaseModel):
    items: List[Dict[str, Any]]
    total: int


class SarSampleAddRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)

    db_id: str
    question: str
    sql: str
    schema_info: Optional[Dict[str, Any]] = Field(default=None, alias="schema")  # 对外仍叫 schema
    source: str = "history"
    verified: bool = True


class SarSampleAddResponse(BaseModel):
    ok: bool
    sample_id: str
    dataset: Dict[str, Any]
