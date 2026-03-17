# backend/app/routers/auth_router.py
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

from app.services.auth_service import login, register_user, verify_token

router = APIRouter(prefix="/auth", tags=["auth"])

class LoginReq(BaseModel):
    account: str
    password: str

class RegisterReq(BaseModel):
    username: str
    contact: str
    password: str

@router.post("/login")
def api_login(req: LoginReq):
    try:
        return login(req.account, req.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/register")
def api_register(req: RegisterReq):
    try:
        return register_user(req.username, req.contact, req.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/me")
def api_me(authorization: str | None = Header(default=None)):
    # 前端 http.js 会带 Authorization: Bearer xxx :contentReference[oaicite:6]{index=6}
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing token")

    token = authorization.split(" ", 1)[1].strip()
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="invalid token")

    return {
        "ok": True,
        "user": {
            "id": int(payload.get("sub")),
            "username": payload.get("username"),
            "role": payload.get("role"),
        }
    }
