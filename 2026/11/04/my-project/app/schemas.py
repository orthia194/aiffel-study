import re
from pydantic import BaseModel, Field, field_validator

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=2,
        max_length=5000,
        description="분석할 한국어 문장 (최소 2자 이상)",
        examples=["배우들 연기도 훌륭하고 스토리 전개도 정말 몰입감 넘치네요!"]
    )

    @field_validator("text")
    @classmethod
    def validate_meaningful_text(cls, v: str) -> str:
        # 1. 앞뒤 공백 제거
        cleaned = v.strip()

        # 2. 최소 2글자 이상인지 체크
        if len(cleaned) < 2:
            raise ValueError("문장의 길이가 너무 짧습니다. 공백을 제외하고 2자 이상 입력해 주세요.")

        # 3. 단순 자음/모음('ㅇ', 'ㅋㅋ')이나 특수문자만 있는지 체크 (완성형 한글 '가-힣' 또는 영문 포함 필수)
        if not re.search(r"[가-힣a-zA-Z]", cleaned):
            raise ValueError("단순 자음/모음이나 기호만으로는 분석할 수 없습니다. 완성된 단어를 포함해 주세요.")

        return cleaned

class PredictResponse(BaseModel):
    success: bool = Field(True, description="성공 여부")
    user: str = Field(..., description="요청한 사용자 이름")
    label: str = Field(..., description="예측 감정 (긍정 / 부정)")
    confidence: float = Field(..., description="예측 신뢰도 점수 (0.0 ~ 1.0)")
    raw_label: str = Field(..., description="모델 원본 라벨 ('0' 또는 '1')")
