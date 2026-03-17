# backend/app/services/auth_service.py
import os
import time
import sqlite3
from typing import Optional, Dict, Any

from passlib.context import CryptContext
from jose import jwt, JWTError

from app.core.config import settings

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
JWT_EXPIRES_SECONDS = int(os.getenv("JWT_EXPIRES_SECONDS", "3600"))

def _users_db_path() -> str:
    # 跟你 history 类似，落在 workspace 里
    return os.path.join(settings.WORKSPACE_DIR, "auth", "users.sqlite3")

def _conn():
    db = _users_db_path()
    os.makedirs(os.path.dirname(db), exist_ok=True)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn

def _ensure_table():
    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            contact TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at INTEGER NOT NULL
        )
        """)
        conn.commit()
    finally:
        conn.close()

def _sign_token(user_id: int, username: str, role: str) -> Dict[str, Any]:
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "iat": now,
        "exp": now + JWT_EXPIRES_SECONDS,
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
    return {"token": token, "expiresIn": JWT_EXPIRES_SECONDS}

def register_user(username: str, contact: str, password: str) -> Dict[str, Any]:
    _ensure_table()
    username = (username or "").strip()
    contact = (contact or "").strip()
    password = (password or "").strip()
    if not username or not contact or not password:
        raise ValueError("missing required fields")

    ph = pwd_ctx.hash(password)

    conn = _conn()
    try:
        cur = conn.cursor()
        # 唯一性检查
        cur.execute("SELECT 1 FROM users WHERE username=? OR contact=?", (username, contact))
        if cur.fetchone():
            raise ValueError("用户名或联系方式已存在")

        cur.execute(
            "INSERT INTO users (username, contact, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (username, contact, ph, "user", int(time.time())),
        )
        conn.commit()
        uid = int(cur.lastrowid)

        signed = _sign_token(uid, username, "user")
        return {
            **signed,
            "user": {"id": uid, "username": username, "role": "user"},
        }
    finally:
        conn.close()

def login(account: str, password: str) -> Dict[str, Any]:
    _ensure_table()
    account = (account or "").strip()
    password = (password or "").strip()
    if not account or not password:
        raise ValueError("账号或密码不能为空")

    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM users WHERE username=? OR contact=?",
            (account, account),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError("账号不存在")

        if not pwd_ctx.verify(password, row["password_hash"]):
            raise ValueError("密码错误")

        uid = int(row["id"])
        username = row["username"]
        role = row["role"]

        signed = _sign_token(uid, username, role)
        return {
            **signed,
            "user": {"id": uid, "username": username, "role": role},
        }
    finally:
        conn.close()

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload
    except JWTError:
        return None
