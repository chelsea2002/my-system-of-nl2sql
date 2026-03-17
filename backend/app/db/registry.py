import os
import json
from typing import Dict, List, Optional
from app.core.config import settings

def _ensure_workspace():
    os.makedirs(settings.WORKSPACE_DIR, exist_ok=True)
    os.makedirs(os.path.join(settings.WORKSPACE_DIR, settings.DBS_DIR), exist_ok=True)
    os.makedirs(os.path.join(settings.WORKSPACE_DIR, settings.SCHEMA_DIR), exist_ok=True)
    os.makedirs(os.path.join(settings.WORKSPACE_DIR, settings.HISTORY_DIR), exist_ok=True)
    os.makedirs(os.path.join(settings.WORKSPACE_DIR, settings.SAR_DIR), exist_ok=True)

def _registry_path() -> str:
    _ensure_workspace()
    return os.path.join(settings.WORKSPACE_DIR, "db_registry.json")

def load_registry() -> Dict[str, Dict]:
    path = _registry_path()
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_registry(reg: Dict[str, Dict]) -> None:
    path = _registry_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)

def register_db(db_id: str, db_path: str) -> None:
    reg = load_registry()
    reg[db_id] = {"db_id": db_id, "db_path": db_path}
    save_registry(reg)

def get_db_path(db_id: str) -> Optional[str]:
    reg = load_registry()
    info = reg.get(db_id)
    return info.get("db_path") if info else None

def list_dbs() -> List[Dict]:
    reg = load_registry()
    out = []
    for db_id, info in reg.items():
        out.append({
            "db_id": db_id,
            "name": db_id,
            "db_path": info.get("db_path")
        })
    return sorted(out, key=lambda x: x["db_id"])
