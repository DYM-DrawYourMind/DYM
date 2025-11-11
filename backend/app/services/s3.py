import uuid
import boto3
from botocore.client import Config
from app import config

def _client():
    return boto3.client(
        "s3",
        endpoint_url=config.S3_ENDPOINT or None,
        aws_access_key_id=config.S3_ACCESS_KEY or None,
        aws_secret_access_key=config.S3_SECRET_KEY or None,
        region_name=config.S3_REGION or None,
        use_ssl=bool(config.S3_USE_SSL),
        config=Config(signature_version="s3v4"),
    )

def presign_put(content_type: str = "image/png", expires_in: int = 600) -> tuple[str, str]:
    """
    S3에 직접 PUT 할 수 있는 presigned URL과 object key를 반환
    """
    key = f"uploads/{uuid.uuid4()}.png"
    url = _client().generate_presigned_url(
        ClientMethod="put_object",
        Params={"Bucket": config.S3_BUCKET, "Key": key, "ContentType": content_type},
        ExpiresIn=expires_in,
    )
    return url, key
