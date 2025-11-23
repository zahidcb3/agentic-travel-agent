import streamlit as st
import uuid
from agents.agent import Agent
from agents.llm_selector import get_llm
from langchain_core.messages import HumanMessage, AIMessage

# MUST be the first Streamlit command
st.set_page_config(page_title="AI Travel Agent", page_icon="🌍", layout="centered")

# Initialize session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    # Stable identifier per Streamlit session for LangGraph checkpointer
    st.session_state.thread_id = str(uuid.uuid4())

# Select and initialize LLM provider (Gemini or Ollama)
_ = get_llm()
st.sidebar.success(f"Model active: {st.session_state.get('active_model_name', 'Unknown')}")

if "agent" not in st.session_state:
    st.session_state.agent = Agent()

st.title("✈️ AI Travel Agent")
st.write("Plan your trips — flights, hotels, and more — with the help of AI!")

# Display chat messages
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# User input box
if user_input := st.chat_input("Ask about flights, hotels, or trip ideas..."):
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            state = {"messages": st.session_state.messages}
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = st.session_state.agent.graph.invoke(state, config=config)
            ai_msg = result["messages"][-1]
            st.markdown(ai_msg.content)
            st.session_state.messages.append(ai_msg)
