from typing import Optional
from fastapi import Depends, HTTPException, Request
from jose import jwt, JWTError
from app import config
from app.db import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models import User


def _extract_token(req: Request) -> Optional[str]:
    # 1) HttpOnly 쿠키 우선
    token = req.cookies.get("access_token")
    if token:
        return token
    # 2) Authorization: Bearer xxx
    auth = req.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return None


async def get_current_user(req: Request, db: AsyncSession = Depends(get_db)):
    token = _extract_token(req)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, config.JWT_SECRET, algorithms=[config.JWT_ALG])
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(status_code=401, detail="Invalid token payload")

        # sub는 환경에 따라 내부 user id 또는 외부 provider id일 수 있음.
        # 우선 숫자형이면 내부 PK(id)로 조회하고, 아니면 kakao_id로 조회 시도.
        user = None
        try:
            # 숫자 문자열이면 내부 id로 조회
            user_id_int = int(sub)
        except Exception:
            user_id_int = None

        if user_id_int is not None:
            res = await db.execute(select(User).where(User.id == user_id_int))
            user = res.scalars().first()

        if user is None:
            # fallback: kakao_id 컬럼으로 조회
            res = await db.execute(select(User).where(User.kakao_id == str(sub)))
            user = res.scalars().first()

        if user:
            return user.to_dict()

        # DB에 없으면 최소한 토큰의 sub만 반환 (호출자에서 처리)
        return {"user_id": sub}
    except JWTError:
        raise HTTPException(status_code=401, detail="Token verification failed")
