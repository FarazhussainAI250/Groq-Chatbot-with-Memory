# groq_chatbot_app.py
# -------------------------------------------------------------
# Streamlit Chatbot using Groq + LangChain (Pro version)
# Features:
# - Chat bubbles UI (Streamlit chat)
# - Conversation memory (Buffer / Summary / Window)
# - Mode switcher for system prompts (Teaching, Coding, Translator, General)
# - Typing effect for assistant replies
# - Download chat history (.txt)
# - Clear chat button
# -------------------------------------------------------------

from dotenv import load_dotenv
import os
import time
import streamlit as st

from langchain_groq import ChatGroq
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
)
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ---------------- Setup & Config ----------------
load_dotenv()

st.set_page_config(page_title="Groq Chatbot with Memory", page_icon="🤖")
st.title("🤖 Groq Chatbot with Memory — Pro")


# --------------- Sidebar Controls ---------------
with st.sidebar:
    st.subheader("⚙️ Controls")
    
    # API Key Input
    GROQ_API_KEY = st.text_input(
        "Please enter your Groq API key:",
        type="password",
        help="Enter your Groq API key to activate the chatbot"
    )

    # Model + generation params
    model_name = st.selectbox(
        "Groq Model",
        [
            "qwen/qwen3-32b",
            "llama-3.3-70b-versatile",
            "deepseek-r1-distill-llama-70b",
            "gemma2-9b-it",
        ],
        index=2,
    )

    temperature = st.slider("Temperature (creativity)", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 50, 3000, 512)

    # Mode switcher → presets for system prompt
    mode = st.selectbox(
        "Assistant Mode",
        ["🎓 Teaching Assistant", "👨‍💻 Coding Helper", "🌍 Translator", "🧠 General Assistant"],
        index=0,
    )

    preset_prompts = {
        "🎓 Teaching Assistant": (
            "You are a helpful, concise teaching assistant. Use short, clear explanations, ideally within 3 lines."
        ),
        "👨‍💻 Coding Helper": (
            "You are a precise coding assistant. Respond with minimal prose, correct code, and brief tips."
        ),
        "🌍 Translator": (
            "You are a professional translator. Preserve meaning and tone. If user doesn't specify, translate to Roman Urdu and English side-by-side."
        ),
        "🧠 General Assistant": (
            "You are a friendly, efficient assistant. Be brief, accurate, and helpful."
        ),
    }

    default_system_prompt = preset_prompts.get(mode, preset_prompts["🧠 General Assistant"]) 
    system_prompt = st.text_area(
        "System Prompt (Rules)",
        value=default_system_prompt,
        help="Edit the assistant's behavior here.",
    )

    st.caption("💡 Tip: Lower temperature for factual tasks; increase for brainstorming.")

    # Memory configuration
    memory_type = st.selectbox(
        "Memory Type",
        ["Buffer (all)", "Summary (long chats)", "Window (last N)"],
        index=0,
        help=(
            "Buffer keeps everything, Summary compresses old context, Window only keeps the last N exchanges."
        ),
    )

    window_k = None
    if memory_type == "Window (last N)":
        window_k = st.slider("Window size (messages)", 2, 20, 6)

    if st.button("🗑️ Clear Chat"):
        if "memory" in st.session_state:
            st.session_state.memory.clear()
        st.success("Chat history cleared!")
        st.rerun()

# --------------- Key Checks ---------------
if not GROQ_API_KEY:
    st.warning("⚠️ Please enter your Groq API key in the sidebar to start chatting.")
    st.stop()

# --------------- Initialize Memory ---------------
# Recreate memory when memory type or window size changes
mem_key = f"memory::{memory_type}::{window_k}"
if "_mem_config" not in st.session_state or st.session_state.get("_mem_config") != mem_key:
    # Create a fresh memory object based on current settings
    if memory_type == "Buffer (all)":
        memory = ConversationBufferMemory(return_messages=True)
    elif memory_type == "Summary (long chats)":
        # Needs an LLM to summarize; use a lightweight Groq model for summaries too
        summarizer_llm = ChatGroq(model_name=model_name, temperature=0, api_key=GROQ_API_KEY)
        memory = ConversationSummaryMemory(llm=summarizer_llm, return_messages=True)
    elif memory_type == "Window (last N)":
        memory = ConversationBufferWindowMemory(k=window_k or 6, return_messages=True)
    else:
        memory = ConversationBufferMemory(return_messages=True)

    st.session_state.memory = memory
    st.session_state._mem_config = mem_key

# --------------- Build LLM ---------------
llm = ChatGroq(
    model_name=model_name,
    temperature=temperature,
    max_tokens=max_tokens,
    api_key=GROQ_API_KEY,
)

# --------------- Conversation Chain ---------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    prompt=prompt,
    verbose=False,
)

# --------------- Helper: Render existing history as chat bubbles ---------------
for msg in st.session_state.memory.chat_memory.messages:
    role = getattr(msg, "type", "ai")  # 'human' or 'ai'
    if role == "human":
        st.chat_message("user").markdown(msg.content)
    else:
        st.chat_message("assistant").markdown(msg.content)

# --------------- Chat Input & Typing Effect ---------------
user_input = st.chat_input("Type your message…")


if user_input:
    # Show user's message bubble immediately
    st.chat_message("user").markdown(user_input)

    # Get assistant full response once, then reveal with typing effect
    full_response = conversation.predict(input=user_input)

    # Typing animation
    placeholder = st.chat_message("assistant").empty()
    accum = ""
    for ch in full_response:
        accum += ch
        placeholder.markdown(accum)
        time.sleep(0.012)  # typing speed

# --------------- Download Chat History ---------------
if st.session_state.memory.chat_memory.messages:
    history_lines = []
    for m in st.session_state.memory.chat_memory.messages:
        role = getattr(m, "type", "ai").upper()
        history_lines.append(f"{role}: {m.content}")
    history_text = "\n\n".join(history_lines)

    st.download_button(
        "📥 Download Chat (.txt)",
        data=history_text,
        file_name="chat_history.txt",
        mime="text/plain",
        help="Save the full conversation as a text file.",
    )

# ---------------- Footer tip ----------------
st.caption(
    "Made with LangChain + Groq. Switch modes and memory types from the sidebar for best results."
)


# ----------- Custom CSS with Header & Footer ------------
st.markdown("""
<style>
/* Background Image */
.stApp {
    background-image: url("https://media.istockphoto.com/id/1488335095/vector/3d-vector-robot-chatbot-ai-in-science-and-business-technology-and-engineering-concept.jpg?s=612x612&w=0&k=20&c=MSxiR6V1gROmrUBe1GpylDXs0D5CHT-mn0Up8D50mr8=");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

/* Top-right Header */
#top-header {
    position: fixed;
    top: 80px;
    right: 20px;
    background-color: rgba(0,0,0,0.5);
    padding: 8px 16px;
    border-radius: 8px;
    color: white;
    font-size: 18px;
    font-weight: bold;
    z-index: 100;
}

/* Bottom-left Footer */
#bottom-footer {
    position: fixed;
    bottom: 10px;
    left: 300px;
    background-color: rgba(0,0,0,0.5);
    padding: 6px 14px;
    border-radius: 6px;
    color: white;
    font-size: 14px;
    z-index: 100;
}
</style>

<div id="top-header">Respected Sir Shahzaib & Sir Ali Hamza</div>
<div id="bottom-footer">Developed by Faraz Hussain</div>
""", unsafe_allow_html=True)
