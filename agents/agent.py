# pylint: disable = http-used,print-used,no-self-use

import datetime
import operator
import os
from typing import Annotated, TypedDict

import streamlit as st

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

from agents.tools.flights_finder import flights_finder
from agents.tools.hotels_finder import hotels_finder
from agents.tools.itinerary_builder import build_itinerary
from agents.llm_selector import get_llm

_ = load_dotenv()

CURRENT_YEAR = datetime.datetime.now().year


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


TOOLS_SYSTEM_PROMPT = f"""You are a smart AI travel planner and assistant.

Your goal is to help the user plan their travel — including flights, hotels, and trip details.

You can ask the user follow-up questions if some details are missing (for example, travel dates, city names, budget, or preferences).

If the user's question is ambiguous, ask questions first — do not make assumptions.

When enough information is available, use your tools to find relevant results.

You have access to three tools:
- `flights_finder`: find flights using Google Flights.
- `hotels_finder`: find hotels using Google Hotels.
- `build_itinerary`: generate detailed itineraries (AI-only; no external APIs).

Tool calling guidance:
- If the user asks for a travel plan or itinerary, call `build_itinerary` with
  {{ "params": {{ "destination": "...", "days": ..., "travelers": ..., "interests": ["..."] }} }}
  Example triggers: "Build a 5-day itinerary for Singapore", "Plan my 3 days in Goa",
  "Give me a detailed Dubai trip plan".

Return your answers in a helpful, conversational format with flight/hotel names, prices, and links where possible.

Always include:
- Airline/hotel name and logo (if possible)
- Prices with currency
- Links to book or view details
- Duration and location info
- Hotel class and ratings (if available)

Current year: {CURRENT_YEAR}
"""

TOOLS = [flights_finder, hotels_finder, build_itinerary]

EMAILS_SYSTEM_PROMPT = """Your task is to convert structured markdown-like travel data into a valid HTML email body.

Rules:
- Do not include any ```html code block preambles.
- Output should be proper HTML ready to be used as email body.
- Make it clean, readable, and visually formatted with headers and lists.

Example format:
<!DOCTYPE html>
<html>
<head><title>Trip Summary</title></head>
<body>
    <h2>Flights</h2>
    <ul>
        <li><strong>Airline:</strong> Emirates — $550 USD</li>
    </ul>
    <h2>Hotels</h2>
    <ul>
        <li><strong>Hotel:</strong> Hilton Paris — $200/night</li>
    </ul>
</body>
</html>
"""


class Agent:
    def __init__(self):
        self._tools = {t.name: t for t in TOOLS}
        # Use the new provider selector (Gemini or Ollama)
        self._get_llm = get_llm
        # Keep both base LLM and tools-bound LLM. Some Ollama models do not
        # support tool calling; we will gracefully fallback at runtime.
        self._base_llm = self._get_llm()
        self._tools_llm = self._base_llm.bind_tools(TOOLS)

        builder = StateGraph(AgentState)
        builder.add_node("call_tools_llm", self.call_tools_llm)
        builder.add_node("invoke_tools", self.invoke_tools)
        builder.add_node("email_sender", self.email_sender)
        builder.set_entry_point("call_tools_llm")

        builder.add_conditional_edges(
            "call_tools_llm",
            Agent.exists_action,
            {"more_tools": "invoke_tools", "email_sender": "email_sender"},
        )
        builder.add_edge("invoke_tools", "call_tools_llm")
        builder.add_edge("email_sender", END)

        memory = MemorySaver()
        self.graph = builder.compile(checkpointer=memory, interrupt_before=["email_sender"])

        print(self.graph.get_graph().draw_mermaid())

    # get_llm is now provided by agents.llm_selector

    @staticmethod
    def exists_action(state: AgentState):
        result = state["messages"][-1]
        if len(result.tool_calls) == 0:
            return "email_sender"
        return "more_tools"

    # 📨 Email sending node
    def email_sender(self, state: AgentState):
        print("Sending email...")
        email_llm = self._get_llm(temperature=0.1)
        email_message = [
            SystemMessage(content=EMAILS_SYSTEM_PROMPT),
            HumanMessage(content=state["messages"][-1].content),
        ]
        email_response = email_llm.invoke(email_message)
        print("Email content generated.")

        message = Mail(
            from_email=os.environ["FROM_EMAIL"],
            to_emails=os.environ["TO_EMAIL"],
            subject=os.environ["EMAIL_SUBJECT"],
            html_content=email_response.content,
        )

        try:
            sg = SendGridAPIClient(os.environ.get("SENDGRID_API_KEY"))
            response = sg.send(message)
            print("Email sent:", response.status_code)
        except Exception as e:
            print("Error sending email:", str(e))

    # 🔍 Call model to decide what to do next
    def call_tools_llm(self, state: AgentState):
        messages = [SystemMessage(content=TOOLS_SYSTEM_PROMPT)] + state["messages"]
        try:
            message = self._tools_llm.invoke(messages)
        except Exception as e:
            # Graceful fallback when the selected model does not support tools
            # (e.g., some Ollama models like llama3:latest).
            err_text = str(e)
            if "does not support tools" in err_text or (
                "tools" in err_text and "support" in err_text
            ):
                st.sidebar.warning(
                    "Selected model does not support tool calling; continuing without tools."
                )
                message = self._base_llm.invoke(messages)
            else:
                raise
        return {"messages": [message]}

    # ⚙️ Run tool calls
    def invoke_tools(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []

        for t in tool_calls:
            print(f"Calling tool: {t['name']}")
            if t["name"] not in self._tools:
                print("⚠️ Invalid tool name. Asking model to retry.")
                result = "Invalid tool name. Please retry."
            else:
                result = self._tools[t["name"]].invoke(t["args"])

            results.append(
                ToolMessage(
                    tool_call_id=t["id"],
                    name=t["name"],
                    content=str(result),
                )
            )

        print("Tool execution complete. Returning to model.")
        return {"messages": results}
