from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.uploads import router as uploads_router
from app.api.v1.auth.kakao.login import router as kakao_login_router
from app.api.v1.auth.kakao.callback import router as kakao_callback_router
from app.api.v1.endpoints.me import router as me_router
from app.api.v1.endpoints.auth import router as auth_router


app = FastAPI(title="ai-art-emotion", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    # 개발 편의: 로컬 프론트 서버(포트 3000/5173)와 127.0.0.1 변형 허용.
    # 배포 시에는 반드시 정확한 origin 목록으로 제한하세요.
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 헬스
app.include_router(health_router, prefix="/health", tags=["health"])
# 업로드
app.include_router(uploads_router, prefix="/api/v1/uploads", tags=["uploads"])
#  카카오 인증 
app.include_router(kakao_login_router,   prefix="/api/v1/auth/kakao", tags=["auth"])
app.include_router(kakao_callback_router, prefix="/api/v1/auth/kakao", tags=["auth"])
# 사용자 인증
app.include_router(me_router, prefix="/api/v1", tags=["auth"])
#토큰
app.include_router(auth_router, prefix="/api/v1")