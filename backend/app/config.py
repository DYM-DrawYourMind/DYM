import os

# Optional: load .env from `backend/.env` if python-dotenv is installed
try:
	from dotenv import load_dotenv
	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	dotenv_path = os.path.join(base_dir, ".env")
	load_dotenv(dotenv_path)
except Exception:
	# python-dotenv not installed or .env missing — environment variables will be read directly
	pass

# --- S3/MinIO 설정 (로컬 개발 시 MinIO 사용 가능) ---
S3_ENDPOINT = os.getenv("S3_ENDPOINT") or None            # 예: "http://localhost:9000" (MinIO), AWS면 None
S3_REGION = os.getenv("S3_REGION", "ap-northeast-2")  # 서울 리전 예시
S3_BUCKET = os.getenv("S3_BUCKET", "uploads")         # 버킷 이름
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "")            # 접근키
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "")            # 비밀키
S3_USE_SSL = bool(os.getenv("S3_USE_SSL", "False") in ("True", "true", "1"))



# --- Kakao OAuth ---
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY", "4e587ae972d2e27da329cc72839081cd")
KAKAO_REDIRECT_URI = os.getenv("KAKAO_REDIRECT_URI", "http://localhost:8000/api/v1/auth/kakao/callback")

# OAuth 고정 엔드포인트 (카카오 표준)
KAKAO_AUTH_AUTHORIZE_URL = "https://kauth.kakao.com/oauth/authorize"
KAKAO_AUTH_TOKEN_URL = "https://kauth.kakao.com/oauth/token"
KAKAO_API_USERINFO_URL = "https://kapi.kakao.com/v2/user/me"

# --- JWT ---
# 프로덕션에서는 반드시 환경변수로 강한 시크릿을 주입하세요.
JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_TO_RANDOM")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", 60 * 24))
COOKIE_NAME = os.getenv("COOKIE_NAME", "access_token")

# --- Database (PostgreSQL async) ---
# 환경변수 `DATABASE_URL`을 우선 사용합니다. 예: postgresql+asyncpg://user:pass@host:5432/dbname
DATABASE_URL = os.getenv(
	"DATABASE_URL",
	"postgresql+asyncpg://dym_user:strongpassword@127.0.0.1:5432/dym_db",
)

# Development helper: allow accepting `state` from query when cookie is missing.
# This should NEVER be enabled in production. Set to 'True' for local testing only.
KAKAO_ALLOW_STATE_IN_QUERY = bool(os.getenv("KAKAO_ALLOW_STATE_IN_QUERY", "False") in ("True", "true", "1"))