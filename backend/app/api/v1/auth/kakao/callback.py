from fastapi import APIRouter, HTTPException, Query, Request, Depends
from fastapi.responses import JSONResponse
import logging
import httpx

from app import config
from app.security import create_access_token  # JWT 생성 유틸
from app.db import get_db
from sqlalchemy import select
from app.models import User
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


@router.get("/callback", summary="Kakao Callback")
async def kakao_callback(
    request: Request,
    code: str = Query(..., description="Authorization code from Kakao"),
    state: str | None = None,
    db: AsyncSession = Depends(get_db),
):

    # Validate state cookie to mitigate CSRF
    cookie_state = request.cookies.get("kakao_oauth_state")
    if not state or not cookie_state or cookie_state != state:
        # Development bypass: allow state in query when cookie is missing if explicit env flag set
        if getattr(config, "KAKAO_ALLOW_STATE_IN_QUERY", False):
            logging.warning("KAKAO_ALLOW_STATE_IN_QUERY enabled: accepting state from query without cookie (dev only).")
        else:
            raise HTTPException(status_code=400, detail="invalid or missing oauth state")

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
        # include provider response for easier debugging
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
        "kakao_id": str(me.get("id")),
        "nickname": (me.get("properties") or {}).get("nickname"),
        "profile_image": (me.get("properties") or {}).get("profile_image"),
        "email": (me.get("kakao_account") or {}).get("email"),
    }

    # 3) DB upsert: find user by kakao_id or create
    stmt = select(User).where(User.kakao_id == profile["kakao_id"])
    res = await db.execute(stmt)
    user = res.scalars().first()
    if not user:
        user = User(kakao_id=profile["kakao_id"], email=profile.get("email"), nickname=profile.get("nickname"))
        db.add(user)
        await db.commit()
        await db.refresh(user)
    else:
        updated = False
        if profile.get("email") and user.email != profile.get("email"):
            user.email = profile.get("email")
            updated = True
        if profile.get("nickname") and user.nickname != profile.get("nickname"):
            user.nickname = profile.get("nickname")
            updated = True
        if updated:
            db.add(user)
            await db.commit()
            await db.refresh(user)

    # 4) JWT 발급 (internal user id)
    user_id = str(user.id)
    token = create_access_token(sub=user_id)

    # 5) HttpOnly 쿠키로 세션 설정 (배포 시 secure=True 권장)
    resp = JSONResponse({"ok": True, "profile": profile, "state": state})
    resp.set_cookie(
        key=config.COOKIE_NAME,
        value=token,
        httponly=True,
        secure=False,           # HTTPS 배포 환경에서는 True 로 변경
        samesite="Lax",        # 프론트 도메인 분리면 "None" + secure=True
        max_age=config.JWT_EXPIRE_MINUTES * 60,
        path="/",
    )
    # remove the temporary oauth state cookie
    resp.delete_cookie("kakao_oauth_state", path="/")
    return resp
