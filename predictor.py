import json
import re
import time
import prompts
import getGeminiResponse as gemini
from task_manager import update_task_progress
from tqdm import tqdm

def predict_batch(task, batch_data):
    """Formats and sends a single batch to Gemini, returning predictions."""
    prompt = prompts.GET_SCORE_PREDICTIONS.replace(
        "{{ $('On form submission').item.json['Task to do'] }}", 
        task
    ).format(json.dumps(batch_data, indent=2))
    
    retries = 3
    while retries > 0:
        try:
            predictions = gemini.getResponse(prompt, json=True, debug=False)
            if isinstance(predictions, list):
                return predictions
            else:
                print(f"Warning: Expected list of predictions, got: {predictions}")
                retries -= 1
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "quota" in err_msg.lower() or "limit" in err_msg.lower():
                delay = 35.0
                match1 = re.search(r"Please retry in ([\d\.]+)s", err_msg)
                match2 = re.search(r"seconds:\s*(\d+)", err_msg)
                if match1:
                    delay = float(match1.group(1)) + 1.0
                elif match2:
                    delay = float(match2.group(1)) + 1.0
                
                print(f"Rate limit / Quota exceeded. Sleeping for {delay:.2f} seconds before retrying...")
                time.sleep(delay)
                retries -= 1
            else:
                print(f"Error getting predictions for batch: {e}")
                break
    return None

def get_predictions_and_save_progress(task, rows_df, taskPath):
    """
    Processes the rows in batches of 10, requests predictions,
    and updates the JSON file after each batch.
    """
    all_predictions = []
    batch_size = 10
    
    print(f"Starting batch prediction for {len(rows_df)} rows...")
    
    # Use tqdm progress bar over the range of batches
    for i in tqdm(range(0, len(rows_df), batch_size), desc="Sending batches to Gemini"):
        batch_df = rows_df.iloc[i : i + batch_size]
        batch_data = batch_df[['row_number', 'text']].to_dict(orient='records')
        
        batch_predictions = predict_batch(task, batch_data)
        
        if batch_predictions:
            all_predictions.extend(batch_predictions)
            # Update the task JSON file with these new predictions incrementally
            update_task_progress(taskPath, batch_predictions)
        else:
            tqdm.write(f"Failed to get predictions for batch starting at index {i}.")
        
        # Rate limit mitigation: wait 2 seconds between batch requests
        time.sleep(2)
            
    print(f"Obtained {len(all_predictions)} predictions.")
    return all_predictions

