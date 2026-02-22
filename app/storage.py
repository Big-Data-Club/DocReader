"""MinIO document storage."""
import io
import json
import uuid
from datetime import datetime
from typing import Optional

from minio import Minio
from minio.error import S3Error

from app.config import settings


def get_minio_client() -> Minio:
    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )


def ensure_bucket():
    client = get_minio_client()
    try:
        if not client.bucket_exists(settings.minio_bucket):
            client.make_bucket(settings.minio_bucket)
    except S3Error as e:
        print(f"MinIO bucket error: {e}")


def upload_document(file_bytes: bytes, filename: str, content_type: str) -> dict:
    """Upload file to MinIO, return doc metadata."""
    ensure_bucket()
    client = get_minio_client()

    doc_id = str(uuid.uuid4())
    ext = filename.rsplit(".", 1)[-1] if "." in filename else "bin"
    object_name = f"{doc_id}/{filename}"

    # Upload file
    client.put_object(
        settings.minio_bucket,
        object_name,
        io.BytesIO(file_bytes),
        length=len(file_bytes),
        content_type=content_type,
    )

    # Upload metadata
    meta = {
        "id": doc_id,
        "filename": filename,
        "content_type": content_type,
        "size": len(file_bytes),
        "uploaded_at": datetime.utcnow().isoformat(),
        "object_name": object_name,
    }
    meta_bytes = json.dumps(meta).encode()
    client.put_object(
        settings.minio_bucket,
        f"{doc_id}/meta.json",
        io.BytesIO(meta_bytes),
        length=len(meta_bytes),
        content_type="application/json",
    )

    return meta


def get_document_bytes(doc_id: str, filename: str) -> bytes:
    client = get_minio_client()
    response = client.get_object(settings.minio_bucket, f"{doc_id}/{filename}")
    return response.read()


def get_document_meta(doc_id: str) -> Optional[dict]:
    client = get_minio_client()
    try:
        response = client.get_object(settings.minio_bucket, f"{doc_id}/meta.json")
        return json.loads(response.read())
    except S3Error:
        return None


def list_documents() -> list[dict]:
    ensure_bucket()
    client = get_minio_client()
    docs = []
    seen = set()
    for obj in client.list_objects(settings.minio_bucket, recursive=True):
        if obj.object_name.endswith("/meta.json"):
            doc_id = obj.object_name.split("/")[0]
            if doc_id not in seen:
                seen.add(doc_id)
                meta = get_document_meta(doc_id)
                if meta:
                    docs.append(meta)
    return docs
