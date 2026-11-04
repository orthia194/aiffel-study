import streamlit as st
import requests

st.set_page_config(
    page_title="한국어 감정 분석 서비스",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 영화/리뷰 감정 분석 서비스")
st.write("문장을 입력하면 AI 모델이 **긍정**인지 **부정**인지 분석해 드립니다.")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ API 접속 설정")
    api_url = st.text_input("백엔드 주소", value="http://localhost:8000")
    api_key = st.text_input("X-API-Key", value="test-key-001", type="password")

    if st.button("서버 상태 확인"):
        try:
            res = requests.get(f"{api_url}/health", timeout=3)
            if res.status_code == 200:
                data = res.json()
                if data.get("model_loaded"):
                    st.success("✅ 서버 및 모델 정상 준비 완료")
                else:
                    st.warning("⚠️ 서버는 동작 중이나 모델 로드 실패")
            else:
                st.warning(f"⚠️ 상태 코드: {res.status_code}")
        except Exception as e:
            st.error(f"❌ 접속 실패: {e}")

# 메인 텍스트 입력창
user_input = st.text_area(
    "분석할 리뷰/문장을 입력하세요:",
    placeholder="예시: 배우들 연기가 정말 좋았고 연출도 최고였습니다!",
    height=120
)

if st.button("감정 분석 시작", type="primary", use_container_width=True):
    # 클라이언트 단에서의 1차 공백 체크
    if not user_input.strip():
        st.warning("⚠️ 공백을 제외한 문장을 최소 1자 이상 입력해 주세요.")
    else:
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }
        payload = {"text": user_input}

        with st.spinner("AI가 분석 중입니다..."):
            try:
                response = requests.post(
                    f"{api_url}/predict",
                    json=payload,
                    headers=headers,
                    timeout=5
                )

                # 1. 성공 (200 OK)
                if response.status_code == 200:
                    data = response.json()
                    label = data["label"]
                    confidence = data["confidence"]
                    user_name = data.get("user", "사용자")

                    st.divider()
                    st.caption(f"🔑 인증 확인: **{user_name}** 님으로 분석되었습니다.")
                    st.subheader("📊 분석 결과")

                    if "긍정" in label:
                        st.success(f"### 🎉 결과: {label}")
                    else:
                        st.error(f"### 💧 결과: {label}")

                    st.write(f"**신뢰도 점수:** `{confidence * 100:.2f}%`")
                    st.progress(confidence)

                # 2. 인증 실패 (401 Unauthorized)
                elif response.status_code == 401:
                    st.error("🔒 인증 실패(401): 유효하지 않은 API Key입니다. 사이드바 설정을 확인하세요.")

                # 3. 입력값 검증 실패 (422 Unprocessable Entity)
                elif response.status_code == 422:
                    error_detail = response.json().get("detail", "입력 데이터 형식이 잘못되었습니다.")
                    st.warning(f"⚠️ 입력 검증 실패(422): {error_detail}")

                # 4. 모델 준비 안 됨 (503 Service Unavailable)
                elif response.status_code == 503:
                    st.error("⏳ 서버는 켜져 있으나 모델이 아직 로드되지 않았습니다(503). 잠시 후 다시 시도해 주세요.")

                # 5. 모델 추론 에러 (500 Internal Server Error)
                elif response.status_code == 500:
                    error_msg = response.json().get("detail", "서버 내부 오류")
                    st.error(f"🚨 모델 추론 중 오류 발생(500): {error_msg}")

                else:
                    st.error(f"기타 오류 발생 (코드: {response.status_code}): {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("🚨 FastAPI 서버에 연결할 수 없습니다. 터미널에서 서버(8000번)가 켜져 있는지 확인하세요.")
            except Exception as e:
                st.error(f"통신 에러 발생: {e}")
