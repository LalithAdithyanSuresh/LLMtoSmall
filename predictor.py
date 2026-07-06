import json
import re
import time
import os
import prompts
import getGeminiResponse as gemini
from task_manager import update_task_progress
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def predict_batch(task, batch_data, api_key=None):
    """Formats and sends a single batch to Gemini, returning predictions."""
    prompt = prompts.GET_SCORE_PREDICTIONS.replace(
        "{{ $('On form submission').item.json['Task to do'] }}", 
        task
    ).format(json.dumps(batch_data, indent=2))
    
    try:
        predictions = gemini.getResponse(prompt, json=True, debug=False, api_key_override=api_key)
        if isinstance(predictions, list):
            return predictions
        else:
            print(f"Warning: Expected list of predictions, got: {predictions}")
    except Exception as e:
        print(f"Error getting predictions for batch: {e}")
    return None


def get_predictions_and_save_progress(task, rows_df, taskPath):
    """
    Processes the rows in batches of 10 concurrently, using all available
    API keys to maximize throughput while respecting the 15 RPM per key rate limit.
    """
    # Load keys
    raw_keys = [
        os.getenv("GEMINI_API_KEY"),
        os.getenv("ALT_GEMINI_API_KEY")
    ]
    api_keys = [k for k in raw_keys if k]
    if not api_keys:
        raise ValueError("No Gemini API keys found in .env file.")
        
    all_predictions = []
    batch_size = 10
    
    # Prepare all batches
    batches = []
    for i in range(0, len(rows_df), batch_size):
        batch_df = rows_df.iloc[i : i + batch_size]
        batch_data = batch_df[['row_number', 'text']].to_dict(orient='records')
        batches.append((i, batch_data))
        
    print(f"Starting concurrent batch prediction for {len(rows_df)} rows using {len(api_keys)} API key(s)...")
    
    # We will submit batches to a thread pool
    # Each worker is assigned an API key round-robin to ensure parallel key usage
    futures = {}
    with ThreadPoolExecutor(max_workers=len(api_keys)) as executor:
        for idx, (batch_start, batch_data) in enumerate(batches):
            assigned_key = api_keys[idx % len(api_keys)]
            
            # Submit to thread pool
            future = executor.submit(predict_batch, task, batch_data, assigned_key)
            futures[future] = batch_start
            
        # Use tqdm to monitor completion
        with tqdm(total=len(batches), desc="Sending batches to Gemini (Concurrent)") as pbar:
            for future in as_completed(futures):
                batch_start = futures[future]
                try:
                    batch_predictions = future.result()
                    if batch_predictions:
                        all_predictions.extend(batch_predictions)
                        # Save progress incrementally to task JSON (thread-safe append)
                        update_task_progress(taskPath, batch_predictions)
                    else:
                        tqdm.write(f"Failed to get predictions for batch starting at index {batch_start}.")
                except Exception as e:
                    tqdm.write(f"Exception raised in batch starting at index {batch_start}: {e}")
                finally:
                    pbar.update(1)
                    
    print(f"Obtained {len(all_predictions)} predictions.")
    return all_predictions
