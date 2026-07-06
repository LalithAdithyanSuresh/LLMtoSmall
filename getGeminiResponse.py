import os
import google.generativeai as genai
from dotenv import load_dotenv
import json as json_lib
import time
import re

# Keep track of the active API key index across calls
current_key_index = 0
last_request_times = {}

def getResponse(prompt, json=True, debug=True, api_key_override=None):
    global current_key_index, last_request_times
    load_dotenv()
    
    # Load all available keys from environment
    raw_keys = [
        os.getenv("GEMINI_API_KEY"),
        os.getenv("ALT_GEMINI_API_KEY")
    ]
    api_keys = [k for k in raw_keys if k]
    
    if api_key_override:
        active_keys = [api_key_override]
        active_index = 0
    else:
        active_keys = api_keys
        active_index = current_key_index
        
    if not active_keys:
        raise ValueError("No Gemini API keys found. Please check your .env file.")
        
    retries = 3
    keys_tried_in_this_attempt = 0
    
    while retries > 0:
        api_key = active_keys[active_index]
        
        # Enforce rate limit (minimum 4.1 seconds between requests on the same key to ensure RPM < 15)
        now = time.time()
        last_time = last_request_times.get(api_key, 0)
        time_elapsed = now - last_time
        min_delay = 4.1  # 4.1 seconds ensures RPM is always strictly under 15
        if time_elapsed < min_delay:
            sleep_time = min_delay - time_elapsed
            time.sleep(sleep_time)
            
        last_request_times[api_key] = time.time()
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3.1-flash-lite')
        
        try:
            response = model.generate_content(prompt)
            break
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "quota" in err_msg.lower() or "limit" in err_msg.lower():
                keys_tried_in_this_attempt += 1
                
                # Switch to alternate key if available and not overridden
                if not api_key_override and len(api_keys) > 1 and keys_tried_in_this_attempt < len(api_keys):
                    print(f"\nRate limit hit on API key index {current_key_index}. Switching to the alternate API key...")
                    current_key_index = (current_key_index + 1) % len(api_keys)
                    active_index = current_key_index
                    continue
                
                # If all available keys are rate limited, sleep and retry
                delay = 35.0
                match1 = re.search(r"Please retry in ([\d\.]+)s", err_msg)
                match2 = re.search(r"seconds:\s*(\d+)", err_msg)
                if match1:
                    delay = float(match1.group(1)) + 1.0
                elif match2:
                    delay = float(match2.group(1)) + 1.0
                
                print(f"\nRate limit / Quota exceeded. Details: {err_msg}")
                print(f"Sleeping for {delay:.2f} seconds before retrying...")
                time.sleep(delay)
                
                # Reset counters for the next retry attempt
                keys_tried_in_this_attempt = 0
                retries -= 1
                if not api_key_override and len(api_keys) > 1:
                    current_key_index = (current_key_index + 1) % len(api_keys)
                    active_index = current_key_index
            else:
                raise e
    else:
        raise Exception("Failed to get response after 3 retries due to quota limits on all keys.")
        
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
