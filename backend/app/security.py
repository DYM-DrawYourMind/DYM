from datetime import datetime, timedelta, timezone
from jose import jwt
from app import config

def create_access_token(sub: str) -> str:
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=config.JWT_EXPIRE_MINUTES)
    payload = {"sub": sub, "iat": int(now.timestamp()), "exp": int(exp.timestamp())}
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALG)
