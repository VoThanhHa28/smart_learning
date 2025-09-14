import os
import firebase_admin
from firebase_admin import credentials, storage

_app_inited = False

def init_firebase():
    global _app_inited
    if _app_inited:
        return
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    bucket_name = os.getenv("FIREBASE_STORAGE_BUCKET")
    if not cred_path or not bucket_name:
        raise RuntimeError("Missing GOOGLE_APPLICATION_CREDENTIALS or FIREBASE_STORAGE_BUCKET")
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})
    _app_inited = True

def download_blob_to_file(storage_path: str, local_path: str) -> str:
    """storage_path ví dụ: 'files/doc1.pdf'"""
    init_firebase()
    bucket = storage.bucket()
    blob = bucket.blob(storage_path)
    if not blob.exists():
        raise FileNotFoundError(f"Blob not found: {storage_path}")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    return local_path
