# app/api/v1/endpoints/auth.py
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt
from app import config

router = APIRouter(prefix="/auth", tags=["auth"])

def _create_access_token(sub: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": sub,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=config.JWT_EXPIRE_MINUTES)).timestamp()),
    }
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALG)

@router.post("/token", summary="아이디/비번으로 JWT 발급(DEV)")
async def issue_token(form: OAuth2PasswordRequestForm = Depends()):
    # DB 없으므로 개발용 간이검증 (원하면 환경변수/설정으로 교체)
    if not (form.username == "admin" and form.password == "admin"):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    token = _create_access_token(sub=form.username)

    # HttpOnly 쿠키로도 심어줌
    resp = Response(content=f'{{"access_token":"{token}","token_type":"bearer"}}',
                    media_type="application/json")
    resp.set_cookie(
        key=config.COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,     # 로컬 http 개발이라 False. 배포시 True(HTTPS)
        max_age=config.JWT_EXPIRE_MINUTES * 60,
    )
    return resp

@router.post("/logout", summary="쿠키 삭제")
async def logout():
    resp = Response(content='{"ok": true}', media_type="application/json")
    resp.delete_cookie(config.COOKIE_NAME)
    return resp
