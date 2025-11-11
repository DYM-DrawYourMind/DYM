# --- S3/MinIO 설정 (로컬 개발 시 MinIO 사용 가능) ---
S3_ENDPOINT = None            # 예: "http://localhost:9000" (MinIO), AWS면 None
S3_REGION = "ap-northeast-2"  # 서울 리전 예시
S3_BUCKET = "uploads"         # 버킷 이름
S3_ACCESS_KEY = ""            # 접근키
S3_SECRET_KEY = ""            # 비밀키
S3_USE_SSL = False            # MinIO http면 False, AWS면 True



# --- Kakao OAuth ---
KAKAO_REST_API_KEY = "4e587ae972d2e27da329cc72839081cd"  # 카카오 Developers > 앱 키 > REST API 키
KAKAO_REDIRECT_URI = "http://localhost:8000/api/v1/auth/kakao/callback"

# OAuth 고정 엔드포인트 (카카오 표준)
KAKAO_AUTH_AUTHORIZE_URL = "https://kauth.kakao.com/oauth/authorize"
KAKAO_AUTH_TOKEN_URL = "https://kauth.kakao.com/oauth/token"
KAKAO_API_USERINFO_URL = "https://kapi.kakao.com/v2/user/me"

# --- JWT ---
JWT_SECRET = "CHANGE_ME_TO_RANDOM"
JWT_ALG = "HS256"
JWT_EXPIRE_MINUTES = 60 * 24
COOKIE_NAME = "access_token"