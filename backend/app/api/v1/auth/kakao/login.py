from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from urllib.parse import urlencode
from secrets import token_urlsafe
from app import config

router = APIRouter()

@router.get("/login")
def kakao_login():
    if not config.KAKAO_REST_API_KEY or not config.KAKAO_REDIRECT_URI:
        raise HTTPException(status_code=500, detail="Kakao OAuth is not configured")
    state = token_urlsafe(16)
    url = (f"{config.KAKAO_AUTH_AUTHORIZE_URL}?"
           + urlencode({
               "client_id": config.KAKAO_REST_API_KEY,
               "redirect_uri": config.KAKAO_REDIRECT_URI,
               "response_type": "code",
               "state": state,
             }))
    return RedirectResponse(url, status_code=302)
