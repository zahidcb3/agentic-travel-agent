import os
from typing import Optional
from datetime import datetime, date

import serpapi
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# from pydantic import BaseModel, Field


class HotelsInput(BaseModel):
    q: str = Field(description='Location of the hotel')
    check_in_date: str = Field(description='Check-in date. The format is YYYY-MM-DD. e.g. 2024-06-22')
    check_out_date: str = Field(description='Check-out date. The format is YYYY-MM-DD. e.g. 2024-06-28')
    sort_by: Optional[str] = Field(8, description='Parameter is used for sorting the results. Default is sort by highest rating')
    adults: Optional[int] = Field(1, description='Number of adults. Default to 1.')
    children: Optional[int] = Field(0, description='Number of children. Default to 0.')
    rooms: Optional[int] = Field(1, description='Number of rooms. Default to 1.')
    hotel_class: Optional[str] = Field(
        None, description='Parameter defines to include only certain hotel class in the results. for example- 2,3,4')


class HotelsInputSchema(BaseModel):
    params: HotelsInput


@tool(args_schema=HotelsInputSchema)
def hotels_finder(params: HotelsInput):
    '''
    Find hotels using the Google Hotels engine.

    Returns:
        dict: Hotel search results.
    '''

    # Basic validation for required fields
    if not params.q or not params.check_in_date or not params.check_out_date:
        return {'error': 'Missing required parameters: q, check_in_date, check_out_date'}

    # Validate date format and chronology
    try:
        ci = datetime.strptime(params.check_in_date, '%Y-%m-%d').date()
        co = datetime.strptime(params.check_out_date, '%Y-%m-%d').date()
    except Exception:
        return {'error': 'Dates must be in YYYY-MM-DD format'}

    today = date.today()
    if ci < today:
        return {'error': '`check_in_date` cannot be in the past.'}
    if co <= ci:
        return {'error': '`check_out_date` must be after `check_in_date`.'}

    query = {
        'api_key': os.environ.get('SERPAPI_API_KEY'),
        'engine': 'google_hotels',
        'hl': 'en',
        'gl': 'us',
        'q': params.q,
        'check_in_date': params.check_in_date,
        'check_out_date': params.check_out_date,
        'currency': 'USD',
        'adults': params.adults,
        'children': params.children,
        'rooms': params.rooms,
        'sort_by': params.sort_by,
        'hotel_class': params.hotel_class
    }

    try:
        search = serpapi.search(query)
        results = search.data
        return results.get('properties', [])[:5]
    except Exception as e:
        # Return a friendly error string so the agent can surface it
        return {'error': str(e)}
