from fastapi import APIRouter, Depends
from deps.auth import get_current_user

router = APIRouter()

@router.get("/me", summary="현재 사용자 프로필(토큰 기반)")
def read_me(user = Depends(get_current_user)):
    # 추후 DB 연동 시 user_id로 실제 프로필 조회
    return {"ok": True, "user": user}
