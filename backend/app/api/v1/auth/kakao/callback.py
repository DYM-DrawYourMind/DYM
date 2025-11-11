from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
import httpx

from app import config
from app.security import create_access_token  # JWT 생성 유틸

router = APIRouter()

@router.get("/callback", summary="Kakao Callback")
async def kakao_callback(
    code: str = Query(..., description="Authorization code from Kakao"),
    state: str | None = None,
):

    data = {
        "grant_type": "authorization_code",
        "client_id": config.KAKAO_REST_API_KEY,
        "redirect_uri": config.KAKAO_REDIRECT_URI,
        "code": code,
    }
    if getattr(config, "KAKAO_CLIENT_SECRET", ""):
        data["client_secret"] = config.KAKAO_CLIENT_SECRET

    async with httpx.AsyncClient(timeout=10.0) as client:
        token_res = await client.post(
            config.KAKAO_AUTH_TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded;charset=utf-8"},
        )
    if token_res.status_code != 200:
        raise HTTPException(
            status_code=token_res.status_code,
            detail=f"token exchange failed: {token_res.text}",
        )

    token_json = token_res.json()
    access_token = token_json.get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="no access_token in response")

    # 2) 사용자 정보 조회
    async with httpx.AsyncClient(timeout=10.0) as client:
        me_res = await client.get(
            config.KAKAO_API_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if me_res.status_code != 200:
        raise HTTPException(
            status_code=me_res.status_code,
            detail=f"user info failed: {me_res.text}",
        )

    me = me_res.json()
    profile = {
        "id": me.get("id"),
        "nickname": (me.get("properties") or {}).get("nickname"),
        "profile_image": (me.get("properties") or {}).get("profile_image"),
        "email": (me.get("kakao_account") or {}).get("email"),
    }

   # 3) JWT 발급 
    user_id = str(profile["id"])
    token = create_access_token(sub=user_id)

    # 4) HttpOnly 쿠키로 세션 설정 (배포 시 secure=True 권장)
    resp = JSONResponse({"ok": True, "profile": profile, "state": state})
    resp.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=False,           # HTTPS 배포 환경에서는 True 로 변경
        samesite="Lax",         # 프론트 도메인 분리면 "None" + secure=True
        max_age=config.JWT_EXPIRE_MINUTES * 60,
        path="/",
    )
    return resp
