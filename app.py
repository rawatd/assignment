import streamlit as st
import asyncio
from typing import List, Dict, Any

# Assuming these imports are from your project
from backend.crew_agents import ContextualRAGCrew
from backend.services.db_connection import setup_pgvector_extension

# --- Page Configuration and Custom CSS for a Professional Dark Theme ---
st.set_page_config(
    page_title="UAE Govt Chat Bot",
    page_icon="🇦🇪",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* Main App Background */
    .stApp {
        background-color: #0d1117; /* GitHub-like dark theme */
        color: #f0f6fc;
        font-family: 'Inter', sans-serif;
    }
    /* Main Content Container */
    .st-emotion-cache-1j01e33 {
        padding: 0 2rem;
    }
    .st-emotion-cache-1wb35e {
        background-color: #161b22;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        padding: 2rem;
    }
    /* Sidebar Styling */
    .st-emotion-cache-121bd9k {
        background-color: #161b22;
        color: #f0f6fc;
        border-right: 1px solid #30363d;
        padding-top: 1rem;
    }
    /* Chat Message Bubbles */
    .st-emotion-cache-1c7icp1 {
        background-color: #21262d;
        color: #f0f6fc;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: transform 0.2s ease-in-out;
    }
    .st-emotion-cache-1c7icp1:hover {
        transform: translateY(-2px);
    }
    /* User Message Bubble */
    .st-emotion-cache-ch5y22.st-emotion-cache-ch5y22 {
        background-color: #313a45;
        border-left: 4px solid #58a6ff;
    }
    /* Assistant Message Bubble */
    .st-emotion-cache-ch5y22:not(.st-emotion-cache-ch5y22) {
        border-left: 4px solid #3fb950;
    }
    /* Buttons and UI elements */
    .stButton>button {
        background-color: #2188ff;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #0077e6;
        box-shadow: 0 2px 8px rgba(0,119,230,0.3);
    }
    .stSelectbox>label, .stTextInput>label, .stMarkdown, .stSubheader, .stHeader, .stHeader h1 {
        color: #f0f6fc;
    }
    .st-emotion-cache-1k041v7 { /* Expander header */
        background-color: #21262d;
        border-radius: 8px;
        padding: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Centered Title and Flag ---
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Flag_of_the_United_Arab_Emirates.svg/1024px-Flag_of_the_United_Arab_Emirates.svg.png", width=80)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #f0f6fc;'>Chat with UAE Govt Chat Bot</h2>", unsafe_allow_html=True)

# --- Caching and Initialization ---
@st.cache_resource
def initialize_rag_crew():
    """Initializes the RAG Crew pipeline once."""
    try:
        setup_pgvector_extension()
        return ContextualRAGCrew()
    except Exception as e:
        st.error(f"Error during RAG Crew initialization: {e}")
        return None

rag_crew_instance = initialize_rag_crew()

if rag_crew_instance is None:
    st.stop()

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ Controls & History")

    # --- Model Selection ---
    st.subheader("AI Model")
    model_options = ["gemma2:2b", "llama3:8b", "mixtral", "gemma:7b"]
    st.selectbox(
        "Choose a model:",
        options=model_options,
        index=model_options.index("gemma2:2b"),
        disabled=True,
        key="model_selector",
        help="The backend is configured to use 'gemma2:2b' for all queries."
    )
    st.info("Currently using **gemma2:2b**")

    st.divider()

    # --- Chat History Management ---
    st.subheader("Chat Actions")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        #st.experimental_rerun()
        st.rerun() 

    st.divider()

    # --- Recent History Viewer ---
    st.subheader("Recent Conversations")
    history_to_display = st.session_state.messages[-4:]
    if history_to_display:
        for i in range(0, len(history_to_display), 2):
            user_msg = history_to_display[i]
            assistant_msg = history_to_display[i+1] if i+1 < len(history_to_display) else None
            
            with st.expander(f"Conversation {i//2 + 1}", expanded=False):
                st.markdown(f"**👤 User:** {user_msg['content']}")
                if assistant_msg:
                    st.markdown(f"**🤖 Assistant:** {assistant_msg['content']}")
    else:
        st.caption("No history available yet.")
        
    st.divider()

# --- Main Chat Interface ---
chat_container = st.container()
with chat_container:
    # Display existing chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])

    # Get new user input
    if prompt := st.chat_input("Ask a question about UAE Govt laws..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.status("Thinking...", expanded=True) as status_message:
                try:
                    status_message.write("Running RAG pipeline...")
                    chat_history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]
                    
                    response_content = asyncio.run(
                        asyncio.to_thread(rag_crew_instance.run_crew, prompt, chat_history)
                    ).raw

                    full_response = response_content
                    status_message.update(label="Processing complete!", state="complete", expanded=False)

                    message_placeholder.markdown(full_response)
                    
                except Exception as e:
                    status_message.update(label="Error occurred", state="error", expanded=False)
                    st.error(f"Error processing prompt: {e}")
                    full_response = "Sorry, an error occurred while processing your request."
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})