from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
from urllib.parse import urlencode
from secrets import token_urlsafe
from app import config

router = APIRouter()


@router.get("/login")
def kakao_login(request: Request, raw: bool = Query(False, description="If true, return the redirect URL as JSON (useful for Swagger)")):
    """
    Redirect user to Kakao authorize URL.
    If `raw=true` is provided, return JSON `{ url: ... }` instead of issuing a 302 redirect.
    """
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

    # Store state in a short-lived cookie to validate callback (CSRF protection)
    if raw or request.headers.get("accept", "").lower().find("application/json") != -1:
        # Return URL as JSON for tools like Swagger that cannot follow cross-origin redirects
        resp = JSONResponse({"url": url, "state": state})
        resp.set_cookie(
            key="kakao_oauth_state",
            value=state,
            httponly=True,
            samesite="Lax",
            secure=False,
            max_age=300,
            path="/",
        )
        return resp

    resp = RedirectResponse(url, status_code=302)
    resp.set_cookie(
        key="kakao_oauth_state",
        value=state,
        httponly=True,
        samesite="Lax",
        secure=False,  # 개발환경: False, 배포 시 True 권장
        max_age=300,
        path="/",
    )
    return resp


@router.get("/login_page")
def kakao_login_page(request: Request):
        """
        Convenience HTML page that sets the oauth state cookie and then redirects the browser to Kakao.
        Open this URL directly in the browser (or click the link returned by Swagger) to perform login.
        """
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

        html = f"""
        <!doctype html>
        <html>
            <head>
                <meta charset="utf-8" />
                <title>Kakao Login</title>
                <script>window.location.href = "{url}";</script>
            </head>
            <body>
                <p>Redirecting to Kakao... If you are not redirected, <a href="{url}">click here</a>.</p>
            </body>
        </html>
        """

        resp = HTMLResponse(content=html)
        resp.set_cookie(
                key="kakao_oauth_state",
                value=state,
                httponly=True,
                samesite="Lax",
                secure=False,
                max_age=300,
                path="/",
        )
        return resp
