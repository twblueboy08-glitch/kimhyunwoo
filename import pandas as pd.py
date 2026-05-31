import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="스팸 분류기 AI", page_icon="🚨", layout="centered")

# 타이틀 및 제작자 각인 영역
st.title("🚨 AI 스팸 메시지 분류기")
st.caption(" **Made by 김현우** | 나이브 베이즈 알고리즘 기반 실시간 스팸 판별 프로그램")
st.markdown("---")

data = {
    'text': [
        '안녕하세요, 오늘 회의 일정 확인 부탁드립니다.',
        '당첨! 무료 쿠폰을 지금 바로 받아가세요!!!',
        '주말에 시간 되시면 같이 저녁 먹을래요?',
        'ㅅr과티ㅂl ㄹㅇ 실화냐?',
        '최저가 보장! 지금 구매하시면 50% 할인 혜택',
        '팀장님, 요청하신 보고서 송부드립니다.',
        '광고) 비밀 보장 급전 필요하신 분 대출 가능',
        '인 스 타에 "여 배우S양" 쳐봐 핸드폰 해킹됐다는 소문 돌던데 인스타에 영 상 올라와 있음.',
        'Google Play 주문 영수증(2026. 4. 25.)',
        '귀하의 환불 요청을 받았습니다.',
        '[광고] (주)OO투자 매니저입니다. 이번 주 급등 확실시되는 히든 종목 선착순 공개',
        '실시간 라이브 카지노 오픈 기념 웰컴 쿠폰 발급 완료 (코드: XXXX)',
        '[Google] 계정 복구 이메일이 성공적으로 변경되었습니다.',
        '[결제완료] 쿠팡 결제 내역 안내 (주문번호: 12345678)',
        '[Notion] 프로젝트 기획서 페이지에 새로운 댓글이 추가되었습니다.',
        '[국세청] 2025년 귀속 종합소득세 신고 안내',
        '일반인 직캠 및 사생활 영상 모음 (주소 변경됨 확인하셈)',
        '바나나TV 이번에 정지 먹은 비제이 유출본 좌표 풀렸다',
        '[축하] 회원님에게 예치된 미수령 보너스 3,000,000원이 소멸 예정입니다. 즉시 출금하세요.',
        '[네이버페이] 주문하신 상품이 배송을 시작했습니다.'
    ],
    'label': [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(df['text'])

model = MultinomialNB()
model.fit(X_train_vectorized, df['label'])

st.sidebar.header("📊 학습 데이터 현황")
st.sidebar.dataframe(df, use_container_width=True)
st.sidebar.caption(f"총 {len(df)}개의 문장으로 알고리즘이 학습되었습니다.")

st.subheader("📝 메시지 분석하기")
user_input = st.text_area(
    "분석하고 싶은 메시지나 댓글을 입력한 후 아래 버튼을 클릭하세요.",
    placeholder="여기에 내용을 입력하세요...",
    height=120
)

if st.button("🔮 실시간 스팸 분석 시작", type="primary"):
    if user_input.strip():
        new_msg_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(new_msg_vectorized)
        
        probabilities = model.predict_proba(new_msg_vectorized)[0]
        spam_prob = probabilities[1] * 100
        ham_prob = probabilities[0] * 100
        
        st.markdown("### 📊 분석 결과")
        
        if prediction[0] == 1:
            st.error(f"🚨 **스팸 메시지입니다.** (스팸 확률: {spam_prob:.1f}%)")
        else:
            st.success(f"✅ **정상 메시지입니다.** (정상 확률: {ham_prob:.1f}%)")
            st.balloons()
            
    else:
        st.warning("내용을 입력해 주세요!")
