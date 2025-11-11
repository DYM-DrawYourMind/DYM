from fastapi import APIRouter, Query
from app.services.s3 import presign_put

router = APIRouter()

@router.get("/presign")
def get_presign(content_type: str = Query("image/png")):
    """
    클라이언트가 S3로 직접 업로드할 수 있도록 Presigned URL 발급
    """
    url, key = presign_put(content_type=content_type)
    return {"url": url, "key": key}

