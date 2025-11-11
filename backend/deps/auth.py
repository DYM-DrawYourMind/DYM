from typing import Optional
from fastapi import Depends, HTTPException, Request
from jose import jwt, JWTError
from app import config

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

def get_current_user(req: Request):
    token = _extract_token(req)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, config.JWT_SECRET, algorithms=[config.JWT_ALG])
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        # 아직 DB가 없으므로 user_id만 반환
        return {"user_id": sub}
    except JWTError:
        raise HTTPException(status_code=401, detail="Token verification failed")
