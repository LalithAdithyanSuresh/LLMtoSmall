import os
import pandas as pd
import json
import getData
import model_trainer
import getGeminiResponse as gemini
import prompts
from task_manager import get_task_details, initialize_task_file, update_task_metrics
from predictor import get_predictions_and_save_progress

def doTask(task, taskFile):
    """Orchestrate the entire process."""
    taskDetails = get_task_details(task, taskFile)
    
    if taskDetails.get("same") == 0:
        # Initialize task config and files
        taskPath, clean_fileName = initialize_task_file(taskDetails, taskFile)
        
        # Retraining loop parameters
        start_row = 0
        offset = 200
        all_rows = pd.DataFrame()
        all_predictions = []
        retrain_needed = True
        max_rows = 1000
        
        while retrain_needed and len(all_predictions) < max_rows:
            print(f"\n--- Retraining Cycle: Total predicted rows so far: {len(all_predictions)} ---")
            print(f"Fetching rows from index {start_row} (count: {offset})...")
            new_rows = getData.fromFile(clean_fileName, startRow=start_row, offset=offset)
            
            if new_rows.empty:
                print("No more rows available in the dataset.")
                break
                
            all_rows = pd.concat([all_rows, new_rows], ignore_index=True)
            
            # Predict new rows incrementally
            new_predictions = get_predictions_and_save_progress(task, new_rows, taskPath)
            all_predictions.extend(new_predictions)
            
            # Merge rows with predictions on row_number
            pred_df = pd.DataFrame(all_predictions)
            pred_df['row_number'] = pred_df['row_number'].astype(int)
            all_rows['row_number'] = all_rows['row_number'].astype(int)
            
            train_df = pd.merge(all_rows, pred_df, on='row_number')
            
            # Train the model
            metrics = model_trainer.train_model(train_df)
            print("Model training complete.")
            
            # Save the final metrics in task JSON file
            update_task_metrics(taskPath, metrics)
            print(f"Task configuration updated at '{taskPath}'.")
            
            # Evaluate metrics with Gemini
            eval_prompt = prompts.EVALUATE_MODEL_METRICS.format(
                json.dumps(metrics, indent=2), 
                len(train_df)
            )
            print("Evaluating model metrics with Gemini...")
            try:
                eval_res = gemini.getResponse(eval_prompt, json=True, debug=False)
                print(f"Evaluation Decision: {json.dumps(eval_res, indent=4)}")
                
                retrain_needed = eval_res.get("retrain", False)
                if retrain_needed:
                    additional = int(eval_res.get("additional_rows", 100))
                    print(f"Retraining requested: adding {additional} more rows.")
                    start_row = len(all_predictions)
                    offset = additional
                else:
                    print("Model performance is satisfactory. Retraining not required.")
            except Exception as e:
                print(f"Failed to evaluate metrics via Gemini: {e}. Stopping retraining loop.")
                retrain_needed = False
                
        print("Pipeline execution complete.")
    else:
        print("Task already exists and is marked as duplicate/same.")

if __name__ == "__main__":
    doTask("Get all spam emails", "temp/emails_shuffled.csv")