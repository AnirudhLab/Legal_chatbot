import streamlit as st
from scripts.query_kb import get_qa_chain

# --- Init ---
st.set_page_config(page_title="Police Chatbot", page_icon="👮‍♂️")
st.title("👮‍♂️ Police Legal Assistant")
st.markdown("Ask legal questions or how to file police complaints. Example: *'My mobile phone was stolen, what should I do?'*")

@st.cache_resource
def load_chain():
    return get_qa_chain()

qa_chain = load_chain()

# --- UI ---
user_input = st.text_input("Enter your query:", placeholder="Type your legal or complaint question here")

if user_input:
    with st.spinner("Getting legal info..."):
        try:
            response = qa_chain.run(user_input)
            st.markdown("### 🧾 Response:")
            st.markdown(response)
        except Exception as e:
            st.error(f"❌ Error: {e}")
