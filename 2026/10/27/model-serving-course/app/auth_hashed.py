"""API Key 인증.

키 원문을 코드에도 서버 키 목록에도 두지 않습니다. 키 목록에는 해시만 있고,
들어온 키를 같은 방식으로 해시로 바꿔 맞춰 봅니다.
"""
import hashlib
import hmac
import json
import os
from pathlib import Path

from fastapi import Header, HTTPException

KEY_FILE = Path(os.environ.get("D49_KEYFILE", "data/api_keys.json"))


def fingerprint(key: str) -> str:
    """키를 해시로 바꿉니다. 같은 키는 항상 같은 해시이고, 되돌릴 수 없습니다."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def load_registry():
    """해시 -> 부르는 쪽 이름. 파일이 없으면 빈 키 목록입니다."""
    if not KEY_FILE.exists():
        return {}
    return json.loads(KEY_FILE.read_text(encoding="utf-8"))


async def verify_api_key(x_api_key: str = Header(default=None)) -> str:
    """X-API-Key 헤더를 검사하고, 통과하면 부르는 쪽 이름을 돌려줍니다.

    FastAPI 가 파라미터 이름 x_api_key 를 헤더 X-API-Key 로 바꿔 읽습니다
    (밑줄은 하이픈으로, 대소문자는 구분하지 않습니다).
    """
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API Key 가 필요합니다. X-API-Key 헤더를 넣어 주십시오",
            headers={"WWW-Authenticate": "X-API-Key"},
        )

    given = fingerprint(x_api_key)
    for known, caller in load_registry().items():
        # 끝까지 비교합니다. 몇 번째 글자에서 틀렸는지가 시간으로 새지 않게 하려는 것입니다
        if hmac.compare_digest(given, known):
            return caller

    raise HTTPException(
        status_code=401,
        detail="등록되지 않은 API Key 입니다",
        headers={"WWW-Authenticate": "X-API-Key"},
    )
