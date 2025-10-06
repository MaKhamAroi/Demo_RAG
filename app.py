import streamlit as st
from query_engine import answer_query

st.title("📖Demo ระบบถามตอบหลักสูตรสาขาวิชาวิทยาการคอมพิวเตอร์")

query = st.text_input("ถามเกี่ยวกับหลักสูตร:", "")

if query:
    with st.spinner("กำลังค้นหา..."):
        answer = answer_query(query)
    st.subheader("คำตอบ:")
    st.write(answer)
