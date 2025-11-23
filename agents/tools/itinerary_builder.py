from typing import Optional, List

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

from agents.llm_selector import get_llm


ITINERARY_SYSTEM_PROMPT = (
    "You are a professional travel planner.\n"
    "Create a detailed, structured itinerary with:\n"
    "- Day-by-day plan\n"
    "- Hour-by-hour schedule\n"
    "- Best time to visit each attraction\n"
    "- Transportation instructions (MRT/Bus/Taxi/Walking)\n"
    "- Food recommendations\n"
    "- Distance/time between places\n"
    "- Opening/closing hours when relevant\n"
    "- Avoid backtracking geographically\n"
    "- Optimize each day logically\n"
    "- Output in clean Markdown with headings, bullet points, time blocks,\n"
    "  attraction lists, food lists, and travel hints.\n"
)


class ItineraryInput(BaseModel):
    destination: str = Field(description="Destination city or region")
    days: int = Field(description="Number of days in the itinerary; must be >= 1")
    travelers: Optional[int] = Field(None, description="Number of travelers")
    interests: Optional[List[str]] = Field(
        None, description="Optional list of interests (e.g., culture, food, nature)"
    )


class ItineraryInputSchema(BaseModel):
    params: ItineraryInput


@tool(args_schema=ItineraryInputSchema)
def build_itinerary(params: ItineraryInput) -> str:
    """
    Generate a detailed AI-only travel itinerary.

    Returns:
        str: A clean Markdown itinerary covering daily schedule, transport, food, distances,
             and optimized route order.
    """

    # Basic validation
    if params.days < 1:
        return "Error: `days` must be >= 1."

    # Prepare the user instruction for the LLM
    interests_text = ", ".join(params.interests) if params.interests else "none specified"
    travelers_text = f"for {params.travelers} travelers" if params.travelers else "for the traveler(s)"

    user_prompt = (
        f"Build a {params.days}-day itinerary for {params.destination} {travelers_text}.\n"
        f"Interests: {interests_text}.\n\n"
        "Requirements:\n"
        "- Provide an hour-by-hour plan for each day.\n"
        "- Include transport instructions (MRT/Bus/Taxi/Walking).\n"
        "- Include food recommendations near attractions.\n"
        "- Provide distance/time between places and maps/distance hints.\n"
        "- Note opening/closing hours where relevant.\n"
        "- Avoid backtracking geographically; optimize route order logically each day.\n"
        "- End each day with a short summary and optional alternatives.\n"
        "- Output in clean Markdown with headings, bullet points, and time blocks.\n"
    )

    # Use the shared Gemini client with a creative but consistent temperature
    llm = get_llm(temperature=0.7)
    messages = [
        SystemMessage(content=ITINERARY_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error generating itinerary: {e}"