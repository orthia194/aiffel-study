from transformers import pipeline

MODEL_NAME = "daekeun-ml/koelectra-small-v3-nsmc"

# 모델의 출력 라벨을 한글로 매핑 (0: 부정, 1: 긍정)
LABEL_MAP = {
    "0": "부정 (Negative)",
    "1": "긍정 (Positive)",
    "LABEL_0": "부정 (Negative)",
    "LABEL_1": "긍정 (Positive)"
}

def load_model():
    """
    Hugging Face pipeline을 이용해 감정 분석 모델을 로드하여 반환합니다.
    """
    classifier = pipeline(
        task="text-classification",
        model=MODEL_NAME,
        device=-1  # CPU 강제 사용
    )
    return classifier

def predict(model, text: str) -> dict:
    """
    텍스트 입력을 받아 감정 분석 추론 결과를 dict 형태로 반환합니다.
    """
    result = model(text)[0]
    raw_label = str(result["label"])
    confidence = float(result["score"])

    label = LABEL_MAP.get(raw_label, raw_label)

    return {
        "success": True,
        "label": label,
        "confidence": round(confidence, 4),
        "raw_label": raw_label
    }
