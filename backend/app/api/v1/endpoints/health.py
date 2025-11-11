# 헬스체크(상태 점검) 전용 라우터입니다.

from typing import Dict
from fastapi import APIRouter

router = APIRouter()

@router.get("/liveness")
def liveness() -> Dict[str, bool]:
    return {"ok": True}

@router.get("/readiness")
def readiness() -> Dict[str, bool]:
    # TODO: DB/Redis/S3 연결 체크 로직 추가 예정
    return {"ok": True}