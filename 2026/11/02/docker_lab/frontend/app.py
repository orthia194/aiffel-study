import os, requests, streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://api:8000")
st.set_page_config(page_title="프라이빗 모델 서빙 대시보드", layout="wide")
st.title("프라이빗 모델 서빙 대시보드 (Docker Compose)")

api_key = st.sidebar.text_input("X-API-Key", type="password", value="modu-private-llm-key")
if st.sidebar.button("서버 헬스체크 (/health)"):
    try:
        hr = requests.get(f"{API_BASE}/health", timeout=3)
        st.sidebar.json(hr.json())
    except Exception as e:
        st.sidebar.error(f"연결 실패: {e}")

text = st.text_area("분석할 문장 입력", value="컨테이너 볼륨 마운트로 가중치를 안전하고 빠르게 서빙합니다.")
if st.button("추론 요청 (/predict)"):
    try:
        r = requests.post(
            f"{API_BASE}/predict",
            json={"text": text},
            headers={"X-API-Key": api_key},
            timeout=5
        )
        st.write(f"HTTP 상태 코드: `{r.status_code}`")
        st.json(r.json())
    except Exception as e:
        st.error(f"API 호출 오류: {e}")
