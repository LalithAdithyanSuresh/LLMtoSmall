import os
import google.generativeai as genai
from dotenv import load_dotenv

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
        return eval(response.text)
    else:
        return response.text

# def getScorePredictions(dataFrame):

