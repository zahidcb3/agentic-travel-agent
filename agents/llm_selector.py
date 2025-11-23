import os
import streamlit as st
from dotenv import load_dotenv


# Load environment variables from .env
_ = load_dotenv()
def get_llm(temperature: float = 0.7):
    """Return an LLM client using Google AI (Gemini) via LangChain.

    Requirements:
    - GOOGLE_API_KEY must be set in the environment.

    Side effects:
    - Writes provider status to Streamlit sidebar.
    - Stores active model name and LLM instance in st.session_state for reuse.
    """

    # Reuse existing instance if already selected at same temperature
    if (
        "llm_instance" in st.session_state
        and st.session_state.get("llm_temperature") == temperature
    ):
        return st.session_state["llm_instance"]

    google_key = os.environ.get("GOOGLE_API_KEY")

    if not google_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Set it in your environment to use Gemini."
        )
    try:
        # Import here to ensure the package is installed and compatible
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

        # Use ONLY supported model identifiers; default to flash-latest
        env_model = os.environ.get("LLM_MODEL", "gemini-flash-latest")
        allowed_models = {
            "gemini-flash-latest",
            "gemini-pro-latest",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        }
        model_name = env_model if env_model in allowed_models else "gemini-flash-latest"

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_key,
            temperature=temperature,
        )

        st.sidebar.success(f"⚡ Using Gemini: {model_name}")
        st.session_state["active_model_name"] = f"Gemini ({model_name})"
        st.session_state["llm_instance"] = llm
        st.session_state["llm_temperature"] = temperature
        return llm
    except Exception as e:
        st.sidebar.error(f"Gemini initialization failed: {e}")
        raise