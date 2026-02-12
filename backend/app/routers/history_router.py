from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.history_service import promote_history_to_sar

router = APIRouter(prefix="/history", tags=["history"])

class PromoteReq(BaseModel):
    history_id: int

@router.post("/promote")
def history_promote(req: PromoteReq):
    try:
        res = promote_history_to_sar(history_id=req.history_id)
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
