import streamlit as st

from rag_pipeline import KenyaTourismRAG


st.set_page_config(page_title="Kenya Tourism RAG", page_icon="🌍", layout="centered")
st.title("Kenya Tourism RAG Assistant")
st.caption("Ask about destinations, seasons, hotels, and activities in Kenya.")

if "rag" not in st.session_state:
    st.session_state.rag = None
    st.session_state.error = None
    try:
        st.session_state.rag = KenyaTourismRAG()
    except Exception as exc:  # noqa: BLE001
        st.session_state.error = str(exc)

query = st.text_input(
    "Your question",
    placeholder="Example: Best places to visit in Kenya in August for a couple?",
)

if st.session_state.error:
    st.error(st.session_state.error)
    st.info("Run `python scripts/ingest.py` first to build the local vector index.")

if st.button("Get Recommendation", type="primary", disabled=st.session_state.rag is None):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            result = st.session_state.rag.ask(query)
        st.subheader("Recommendation")
        st.write(result["answer"])

        st.subheader("Sources")
        if result["sources"]:
            for src in result["sources"]:
                st.markdown(f"- {src}")
        else:
            st.write("No sources found.")
