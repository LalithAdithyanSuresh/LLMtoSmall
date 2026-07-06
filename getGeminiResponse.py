import os
import google.generativeai as genai
from dotenv import load_dotenv
import json as json_lib

def getResponse(prompt,json=True,debug=True):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Please check your .env file.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.1-flash-lite')
    response = model.generate_content(prompt)
    if debug:
        print(response.text)
    if json:
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        try:
            return json_lib.loads(text)
        except Exception:
            try:
                return eval(text)
            except Exception as e:
                print(f"Failed to parse JSON response: {text}")
                raise e
    else:
        return response.text

# def getScorePredictions(dataFrame):

