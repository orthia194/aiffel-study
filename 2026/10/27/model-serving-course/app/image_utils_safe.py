"""9.2절 image_utils.py 앞에 픽셀 수 검사를 하나 더 둡니다. 원본 파일은 고치지 않습니다."""
from fastapi import HTTPException, UploadFile
from PIL import Image

from app import image_utils

MAX_PIXELS = 4096 * 4096   # 약 1,680만 픽셀. 휴대폰 사진(약 1,200만)은 지나갑니다


async def validate_and_read_image(file: UploadFile, max_pixels: int = MAX_PIXELS, **kwargs):
    """헤더에 적힌 픽셀 수부터 보고, 통과하면 9.2절의 네 단계에 그대로 넘깁니다."""
    try:
        with Image.open(file.file) as head:    # 헤더만 읽고 아직 펼치지 않습니다
            width, height = head.size
    except Exception:
        width = height = 0                      # 열리지 않는 파일은 9.2절 3단계가 400으로 막습니다
    await file.seek(0)                          # 읽은 자리를 처음으로 되돌립니다

    if width * height > max_pixels:
        raise HTTPException(
            status_code=400,
            detail=f"이미지가 너무 큽니다: {width}x{height}. 최대 {max_pixels:,}픽셀까지 받습니다.",
        )
    return await image_utils.validate_and_read_image(file, **kwargs)
