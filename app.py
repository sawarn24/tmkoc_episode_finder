import streamlit as st
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Use the new HuggingFaceEndpointEmbeddings
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",  # Use a proper embedding model
    task="feature-extraction",
    huggingfacehub_api_token=HF_TOKEN
)
st.title("LET'S FIND YOUR TMOCKOC EPISODE")

st.title("🎬 LET'S FIND YOUR TMKOC EPISODE 🎭")
st.write("Got a storyline in mind but can't recall the episode? 🤔")
st.write("Don't worry! Just describe the **main story** or **key events** of the episode (not dialogues).")


user_input=st.text_input("Write whatever you remembered in english")
vectorstore= FAISS.load_local("episodes_faiss", embeddings, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
if user_input:
    results = retriever.get_relevant_documents(user_input)
    st.subheader("🔹 Top Episode Suggestions:")
    
    for i, doc in enumerate(results, start=1):
        st.markdown(f"### 🎥 Suggestion {i}")
        st.write(f"**Episode Number:** {doc.metadata.get('episode_number', 'N/A')}")
        st.write(f"**Title:** {doc.metadata.get('title', 'N/A')}")
        st.write(f"**Description:** {doc.metadata.get('description', 'N/A')}")

        st.markdown("---")
