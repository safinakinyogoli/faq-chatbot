
import os
import streamlit as st
from dotenv import load_dotenv
from rag_engine import RAGEngine

load_dotenv()

st.set_page_config(
    page_title="NHIF Healthcare Chatbot",
    page_icon=":hospital:",
    layout="wide"
)

st.title("NHIF Healthcare FAQ Chatbot")
st.markdown("Welcome! Ask me anything about NHIF healthcare services.")

@st.cache_resource
def init_rag_engine():
    try:
        return RAGEngine()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = init_rag_engine()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about NHIF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if st.session_state.rag_engine:
                response = st.session_state.rag_engine.get_response(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error("Chatbot not available")

with st.sidebar:
    st.header("About")
    st.markdown("Healthcare FAQ Chatbot using RAG")
    if not os.getenv("HUGGINGFACE_API_KEY"):
        st.warning("API Key missing")
