import os
import streamlit as st
from dotenv import load_dotenv
import json

from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the LLM\llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Base prompt for Q&A
prompt = ChatPromptTemplate.from_template(
    """
You are a skilled AI.
You answer ONLY from the given context.
If the answer isn’t there, say "Sorry that is not mentioned in your data."
However answer basic hello/hi questions as a normal LLM.

<context>
{context}
</context>
Question: {input}
"""
)

# Sample Documents for Flashcards and QA
documents = [
    Document(page_content=(
        "The water cycle, also known as the hydrological cycle, describes how water moves "
        "through Earth's atmosphere, land, and oceans. It includes processes like evaporation, "
        "condensation, precipitation, infiltration, runoff, and transpiration. Solar energy drives "
        "the cycle by heating water in rivers, lakes, and oceans, causing it to evaporate into water vapor. "
        "This vapor rises and cools, forming clouds through condensation. When the clouds become heavy, "
        "they release moisture as precipitation. Some of the water infiltrates the soil, replenishing "
        "groundwater, while the rest flows over land as runoff, eventually returning to oceans or lakes. "
        "Plants also release water vapor through transpiration. This continuous movement of water supports "
        "ecosystems, regulates climate, and maintains life on Earth."
    )),
]

# Streamlit Application
st.set_page_config(page_title="🧠 Second Brain", layout="wide")
st.title("🧠 Your Second Brain")

# Function to build vector database
def create_vector_db(docs):
    if "vector_store" not in st.session_state:
        st.session_state.embedding = OllamaEmbeddings(model="gemma:2b")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.split_docs = st.session_state.text_splitter.split_documents(docs)
        st.session_state.vector_store = FAISS.from_documents(
            st.session_state.split_docs,
            st.session_state.embedding
        )

# Define tabs
 tab_ask, tab_history, tab_flash = st.tabs(["💬 Ask", "📜 History", "⚡ Flash Cards"])

with tab_ask:
    if st.button("🔗 Build Vector DB"):
        create_vector_db(documents)
        st.success("✅ Vector store is ready!")

    user_prompt = st.text_input("❓ Ask me anything from your documents:")
    if user_prompt:
        if "vector_store" not in st.session_state:
            st.warning("Please click “Build Vector DB” first.")
        else:
            retriever = st.session_state.vector_store.as_retriever()
            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            result = retrieval_chain.invoke({"input": user_prompt})
            answer = result.get("answer", "No answer returned.")

            st.markdown("**Answer:**")
            st.write(answer)

            # Save to history
            st.session_state.setdefault("history", []).append({
                "user": user_prompt,
                "bot": answer
            })

with tab_history:
    history = st.session_state.get("history", [])
    if not history:
        st.info("No questions asked yet.")
    else:
        for msg in history:
            st.markdown(f"**You:** {msg['user']}")
            st.markdown(f"**Bot:** {msg['bot']}")
            st.divider()

with tab_flash:
    # Flashcard prompt
    flash_prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant that generates study flashcards.
Given the passage below, output a JSON array of objects with `question` and `answer` fields.
Only return valid JSON.

<context>
{context}
</context>
"""
    )
    if "vector_store" not in st.session_state:
        create_vector_db(documents)
    retriever = st.session_state.vector_store.as_retriever()
    document_chain = create_stuff_documents_chain(llm, flash_prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    with st.spinner("Generating flashcards…"):
        result = retrieval_chain.invoke({"input": "Generate 5 flashcards"})
        raw = result.get("answer", "")

    try:
        cards = json.loads(raw)
    except json.JSONDecodeError:
        st.error("❌ Failed to parse JSON. Model returned:")
        st.code(raw)
        st.stop()

    if not cards:
        st.info("No flashcards generated. Try clicking “Build Vector DB” and then reload.")
    else:
        for idx, card in enumerate(cards, start=1):
            q = card.get("question", "❓")
            a = card.get("answer", "❓")
            with st.expander(f"Card {idx}: {q}"):
                st.write(a)
