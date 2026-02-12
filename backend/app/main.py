from dotenv import load_dotenv
load_dotenv() 

import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any

from app.core.config import settings
from app.db.registry import register_db, get_db_path, list_dbs
from app.models.schemas import (
    UploadResponse,
    ParseSchemaRequest,
    SchemaOverview,
    EnrichSchemaRequest,
    EnrichSchemaResponse,
    GenerateRequest,
    GenerateResponse,
    ExecuteRequest,
    HistoryStatsResponse,
    ExecuteResponse,
    FeedbackRequest,
    FeedbackResponse,
    HistoryListResponse,
    SarSampleAddRequest,
    SarSampleAddResponse,
)
from app.services.schema_service import parse_and_persist_schema, enrich_schema, load_schema_dict, load_schema_text
from app.services.final_generate_service import generate_for_api
from app.services.history_service import add_feedback, list_history, get_detail, get_history_stats
from app.services.sar_service import add_sample, import_supervised_file, build_index, get_index_status, add_sample, list_samples, delete_sample, get_dataset_summary
from pydantic import BaseModel
from app.services.sar_runtime import SARRuntime
from app.services.sar_index_service import build_index, get_index_status
from app.services.nl2sql_service import execute as exec_sql
from app.services.history_feedback_service import (
    append_history_record,
    upsert_supervised_sample,
    mark_sar_index_dirty,
)
from app.services.history_store import HistoryStore
from app.routers.history_router import router as history_router
from app.routers.auth_router import router as auth_router

HISTORY_DB = os.path.join(settings.WORKSPACE_DIR, "history", "history.sqlite3")
history_store = HistoryStore(HISTORY_DB)

app = FastAPI(title="NL2SQL Backend", version="0.1.0")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 或 ["*"] 先跑通
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

SAR_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "sar")
# 解释：__file__ = backend/app/main.py
# dirname(dirname(__file__)) = backend/
# => backend/assets/sar

from app.services.schema_link_service import SchemaLinkService

schema_linker = None

@app.on_event("startup")
def _startup():
    global schema_linker
    schema_linker = SchemaLinkService(
        llm_path="/home/super/桌面/SchemaRAG/backend/assets/TonyTANG11/SchemaLinker",  
        emb_path="/home/super/桌面/SchemaRAG/backend/assets/sar/bge-large-en-v1.5",
        device="cuda",
    )


@app.on_event("startup")
def load_sar_runtime():
    try:
        app.state.sar = SARRuntime(
            assets_dir=SAR_ASSETS_DIR,
            device=None,      # 自动 cuda/cpu
            use_fp16=True
        )
        print(f"[SAR] runtime loaded. device={app.state.sar.device}")
    except Exception as e:
        app.state.sar = None
        print(f"[SAR] runtime load failed: {e}")

# CORS：允许你的前端 dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _db_dir(db_id: str) -> str:
    d = os.path.join(settings.WORKSPACE_DIR, settings.DBS_DIR, db_id)
    os.makedirs(d, exist_ok=True)
    return d

@app.get("/health")
def health():
    return {"ok": True}

# ============ DB list ============
@app.get("/db/list")
def db_list():
    return list_dbs()

# ============ Schema upload/parse/enrich ============
@app.post("/schema/upload", response_model=UploadResponse)
async def schema_upload(db_id: str = Form(...), file: UploadFile = File(...)):
    if not db_id.strip():
        raise HTTPException(status_code=400, detail="db_id is required")
    if not file.filename.lower().endswith(".sqlite"):
        raise HTTPException(status_code=400, detail="only .sqlite is supported")

    target_dir = _db_dir(db_id)
    target_path = os.path.join(target_dir, f"{db_id}.sqlite")

    try:
        with open(target_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to save file: {e}")

    register_db(db_id, target_path)
    return UploadResponse(db_id=db_id, db_path=target_path)

@app.post("/schema/parse", response_model=SchemaOverview)
def schema_parse(req: ParseSchemaRequest):
    db_path = get_db_path(req.db_id)
    if not db_path or not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="db not found, please upload first")

    info = parse_and_persist_schema(req.db_id, db_path, k_examples=req.k_examples)  # ✅ 关键
    return SchemaOverview(**info)

@app.post("/schema/enrich", response_model=EnrichSchemaResponse)
def schema_enrich(req: EnrichSchemaRequest):
    db_path = get_db_path(req.db_id)
    if not db_path or not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="db not found, please upload first")

    # 需要先 parse 才有 schema.txt / schema.json
    try:
        result = enrich_schema(req.db_id, db_path, req.question, req.evidence, req.k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return EnrichSchemaResponse(
        db_id=req.db_id,
        schema_text_enriched=result["schema_text_enriched"],
        sample_text=result["sample_text"],
    )

# ============ NL2SQL generate/execute ============
@app.post("/nl2sql/generate", response_model=GenerateResponse)
def nl2sql_generate(req: GenerateRequest):
    db_path = get_db_path(req.db_id)
    if not db_path:
        raise HTTPException(status_code=404, detail="db not found")

    res = generate_for_api(
        db_id=req.db_id,
        question=req.question,
        options=req.options or {},
        schema_linker=schema_linker,   # ✅ 关键
    )
    return GenerateResponse(**res)


@app.post("/nl2sql/execute", response_model=ExecuteResponse)
def nl2sql_execute(req: ExecuteRequest):
    db_path = get_db_path(req.db_id)
    if not db_path or not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="db not found")

    result = exec_sql(db_path, req.sql)
    return ExecuteResponse(**result)

# ============ History ============
@app.get("/history", response_model=HistoryListResponse)
def history_list(db_id: str = "", q: str = "", page: int = 1, page_size: int = 10):
    res = list_history(db_id=db_id, q=q, page=page, page_size=page_size)
    return HistoryListResponse(**res)

@app.get("/history/stats", response_model=HistoryStatsResponse)
def history_stats(db_id: str = ""):
    res = get_history_stats(db_id=db_id)
    return HistoryStatsResponse(**res)

@app.get("/history/{hid}")
def history_detail(hid: str):
    item = get_detail(hid)
    if not item:
        raise HTTPException(status_code=404, detail="not found")
    return item

@app.post("/history/feedback", response_model=FeedbackResponse)
def history_feedback(req: FeedbackRequest):
    payload = req.model_dump()   # pydantic v2；如果是 v1 用 req.dict()
    out = add_feedback(payload)  # 你的 history_service.add_feedback(payload)
    return FeedbackResponse(ok=True, history_id=out["id"] if isinstance(out, dict) else out)

# ============ SAR samples ============
@app.post("/sar/samples/add", response_model=SarSampleAddResponse)
def sar_sample_add(req: SarSampleAddRequest):
    try:
        res = add_sample(
            db_id=req.db_id,
            question=req.question,
            sql=req.sql,
            schema=req.schema_info,
            source=req.source,
            verified=req.verified
        )
        return SarSampleAddResponse(ok=True, sample_id=res["sample_id"], dataset=res["dataset"])
    except ValueError as e:
        if str(e) == "DUPLICATE":
            raise HTTPException(status_code=409, detail="DUPLICATE")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/schema/{db_id}/json")
def get_schema_json(db_id: str):
    try:
        return {
            "db_id": db_id,
            "schema_dict": load_schema_dict(db_id),
            "schema_text": load_schema_text(db_id),
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="schema not parsed")

class SarRetrieveRequest(BaseModel):
    db_id: str
    question: str
    k: int = 5

@app.post("/sar/retrieve")
def sar_retrieve(req: SarRetrieveRequest):
    sar = getattr(app.state, "sar", None)
    if sar is None:
        raise HTTPException(status_code=503, detail="SAR runtime not loaded, check backend/assets/sar")

    # 用你已经持久化的 schema_dict（跟你系统一致）
    try:
        schema = load_schema_dict(req.db_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="schema not parsed, please call /schema/parse first")

    items = sar.retrieve(req.question, schema, k=req.k)
    # 给前端最常用的字段
    return {"db_id": req.db_id, "items": [{"score": x["score"], "question": x["question"], "sql": x["sql"]} for x in items]}

@app.post("/sar/datasets/import")
async def sar_import_dataset(
    db_id: str = Form(...),
    mode: str = Form("merge"),
    file: UploadFile = File(...),
):
    if mode not in ("merge", "replace"):
        raise HTTPException(status_code=400, detail="mode must be merge/replace")

    try:
        text = (await file.read()).decode("utf-8", errors="ignore")
        res = import_supervised_file(db_id=db_id, file_text=text, mode=mode, source="import")
        return res
    except ValueError as e:
        if str(e) == "EMPTY_FILE":
            raise HTTPException(status_code=400, detail="EMPTY_FILE")
        if str(e) == "PARSE_ERROR":
            raise HTTPException(status_code=400, detail="PARSE_ERROR")
        raise HTTPException(status_code=400, detail=str(e))
    

@app.post("/sar/index/build")
def sar_index_build(payload: dict = Body(...)):
    db_id = (payload.get("db_id") or "").strip()
    mode = (payload.get("mode") or "incremental").strip()

    if not db_id:
        raise HTTPException(status_code=400, detail="db_id is required")

    # 如果你希望必须先有 supervised_data 才允许 build，可在这里做校验
    try:
        idx = build_index(db_id=db_id, mode=mode)
        return {"ok": True, "index": idx}
    except ValueError as e:
        if str(e) == "BAD_MODE":
            raise HTTPException(status_code=400, detail="mode must be incremental/full")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/sar/index/status")
def sar_index_status(db_id: str):
    if not db_id.strip():
        raise HTTPException(status_code=400, detail="db_id is required")
    return get_index_status(db_id)

@app.get("/sar/samples")
def sar_samples_list(
    db_id: str,
    q: str = "",
    page: int = 1,
    page_size: int = 10
):
    if not db_id.strip():
        raise HTTPException(status_code=400, detail="db_id is required")
    return list_samples(db_id=db_id, q=q, page=page, page_size=page_size)


@app.delete("/sar/samples")
def sar_samples_delete(payload: dict = Body(...)):
    db_id = (payload.get("db_id") or "").strip()
    sample_id = (payload.get("sample_id") or "").strip()

    if not db_id:
        raise HTTPException(status_code=400, detail="db_id is required")
    if not sample_id:
        raise HTTPException(status_code=400, detail="sample_id is required")

    res = delete_sample(db_id=db_id, sample_id=sample_id)
    if not res.get("ok"):
        raise HTTPException(status_code=404, detail="sample not found")
    return res

@app.get("/sar/datasets/summary")
def sar_dataset_summary(db_id: str):
    db_id = (db_id or "").strip()
    if not db_id:
        raise HTTPException(status_code=400, detail="db_id is required")
    return get_dataset_summary(db_id)

class SarIndexBuildRequest(BaseModel):
    db_id: str
    mode: str = "incremental"

class SarIndexStatusResponse(BaseModel):
    db_id: str
    status: str
    total_embeddings: int
    updated_at: str
    last_build_mode: str
    embedding_model: Optional[str] = None
    dim: Optional[int] = None

@app.post("/sar/index/build")
def sar_index_build(req: SarIndexBuildRequest):
    return build_index(req.db_id, mode=req.mode)

@app.get("/sar/index/status", response_model=SarIndexStatusResponse)
def sar_index_status(db_id: str):
    return get_index_status(db_id)


class FeedbackReq(BaseModel):
    db_id: str
    question: str
    selected_sql: str
    ok: bool
    corrected_sql: Optional[str] = ""

@app.post("/history/feedback")
def history_feedback(req: FeedbackReq):
    db_id = req.db_id
    q = req.question
    sql = req.corrected_sql.strip() if (req.ok is False and req.corrected_sql) else req.selected_sql

    # 1) 写历史（无论 ok/不 ok 都写）
    append_history_record(db_id, {
        "db_id": db_id,
        "question": q,
        "selected_sql": req.selected_sql,
        "ok": req.ok,
        "corrected_sql": req.corrected_sql or "",
        "created_at": int(__import__("time").time() * 1000),
    })

    saved_supervised = False
    reindex_triggered = False

    # 2) 若 ok=true：写入 supervised + 触发 SAR 增量索引
    if req.ok is True:
        schema_snapshot = load_schema_dict(db_id)  # schema snapshot
        saved_supervised = upsert_supervised_sample(
            db_id=db_id,
            question=q,
            sql=sql,
            schema_snapshot=schema_snapshot,
        )
        if saved_supervised:
            mark_sar_index_dirty(db_id)
            reindex_triggered = True

    return {
        "ok": True,
        "saved_history": True,
        "saved_supervised": saved_supervised,
        "reindex_triggered": reindex_triggered,
        "message": "feedback saved"
    }


@app.get("/history")
def get_history(
    db_id: Optional[str] = None,
    q: Optional[str] = None,
    ok: Optional[bool] = None,
    imported: Optional[bool] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
):
    total, items = history_store.list(
        db_id=db_id, q=q, ok=ok, imported=imported, page=page, page_size=page_size
    )
    return {
        "ok": True,
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": items
    }

app.include_router(history_router)



