import os
import pandas as pd
import json
import getData
import model_trainer
import getGeminiResponse as gemini
import prompts
from task_manager import get_task_details, initialize_task_file, update_task_metrics, load_existing_task, save_model
from predictor import get_predictions_and_save_progress

def doTask(task, taskFile):
    """Orchestrate the entire process."""
    taskDetails = get_task_details(task, taskFile)
    
    taskPath = None
    clean_fileName = None
    all_predictions = []
    all_predictions_loaded_at_startup = 0
    model = None
    
    # Check if the task already exists and we need to load its state
    if taskDetails.get("same") == 1:
        taskId = taskDetails.get("taskId")
        taskPath, taskContent = load_existing_task(taskId)
        if taskContent:
            print(f"Task already exists. Continuing from config: {taskPath}")
            clean_fileName = taskContent["fileName"]
            all_predictions = taskContent["rows_used"]
            all_predictions_loaded_at_startup = len(all_predictions)
            
            # Load pretrained model if it exists and matches prediction count
            model_path = taskContent.get("model_path")
            if model_path and os.path.exists(model_path):
                expected_filename = f"{taskId}_{taskDetails['taskName'].replace(' ', '-')}_{len(all_predictions)}.joblib"
                if os.path.basename(model_path) == expected_filename:
                    try:
                        import joblib
                        model = joblib.load(model_path)
                        print(f"Loaded pretrained model for {len(all_predictions)} rows from: {model_path}")
                    except Exception as e:
                        print(f"Failed to load pretrained model: {e}")
                else:
                    print("Pretrained model exists, but its size does not match current prediction count. Bypassing load.")
        else:
            print("Warning: Task marked as existing, but configuration file was not found. Treating as new.")
            
    if not taskPath:
        # Initialize new task config and files (or reuse if file already exists)
        taskPath, clean_fileName, loaded_preds = initialize_task_file(taskDetails, taskFile)
        if loaded_preds:
            all_predictions = loaded_preds
            all_predictions_loaded_at_startup = len(all_predictions)
            
            # Try to load existing model from existing config
            try:
                with open(taskPath, "r") as f:
                    existing_content = json.load(f)
                model_path = existing_content.get("model_path")
                if model_path and os.path.exists(model_path):
                    expected_filename = f"{taskDetails['taskId']}_{taskDetails['taskName'].replace(' ', '-')}_{len(all_predictions)}.joblib"
                    if os.path.basename(model_path) == expected_filename:
                        import joblib
                        model = joblib.load(model_path)
                        print(f"Loaded pretrained model for {len(all_predictions)} rows from: {model_path}")
            except Exception:
                pass
         
    # Retraining loop parameters
    # We aim to reach target_rows_count predictions.
    # If starting fresh, target is 200. If resuming, force adding 300 more rows.
    target_rows_count = len(all_predictions) + 300 if len(all_predictions) > 0 else 200
    
    # If we already have a model matching the target row count, we can skip the loop
    if model is not None and len(all_predictions) >= target_rows_count:
        retrain_needed = False
        print("Pretrained model matching target row count loaded. Skipping retraining loop.")
    else:
        retrain_needed = True
        
    max_rows = 2000
    eval_res = {}
    
    while retrain_needed and len(all_predictions) < max_rows:
        # If we have predictions, but fewer than target_rows_count, predict the missing ones.
        if len(all_predictions) < target_rows_count:
            print(f"\n--- Retraining Cycle: Target predicted rows: {target_rows_count} (Currently have: {len(all_predictions)}) ---")
            
            # Load the full cleaned dataset
            dataset_path = os.path.join("dataPool", clean_fileName)
            df_full = pd.read_csv(dataset_path)
            
            # Filter out already predicted rows
            predicted_ids = {int(p['row_number']) for p in all_predictions}
            
            if len(all_predictions) == 0:
                # No baseline model exists yet; fetch initial baseline rows sequentially
                print(f"No baseline labels found. Fetching first {target_rows_count} rows sequentially for initial training...")
                unpredicted_df = df_full[~df_full['row_number'].astype(int).isin(predicted_ids)].head(target_rows_count)
            else:
                # Active Learning / Uncertainty Sampling:
                print("Running uncertainty sampling to select lowest-confidence rows...")
                # 1. Train a temporary model on current predictions (or reuse loaded startup model if size matches)
                if model is not None and len(all_predictions) == all_predictions_loaded_at_startup:
                    print("Reusing pretrained model loaded at startup for uncertainty sampling.")
                    temp_model = model
                else:
                    temp_train_df = pd.merge(df_full, pd.DataFrame(all_predictions), on='row_number')
                    temp_model, _ = model_trainer.train_model(temp_train_df)
                
                # 2. Predict confidence on all unpredicted rows in the dataset
                unpredicted_df = df_full[~df_full['row_number'].astype(int).isin(predicted_ids)].copy()
                y_pred, confidence = model_trainer.predict_with_confidence(temp_model, unpredicted_df['text'])
                unpredicted_df['confidence'] = confidence
                
                # 3. Sort by confidence ascending (lowest first) and pick the top missing ones
                unpredicted_df = unpredicted_df.sort_values(by='confidence', ascending=True).head(target_rows_count - len(all_predictions))
                print(f"Selected {len(unpredicted_df)} low-confidence rows (Avg confidence of selection: {unpredicted_df['confidence'].mean():.4f})")
                
                # Drop temporary confidence column before passing to predictor
                unpredicted_df = unpredicted_df.drop(columns=['confidence'])
            
            if unpredicted_df.empty:
                print("No more rows available in the dataset.")
                break
                
            print(f"Need predictions for {len(unpredicted_df)} new rows.")
            new_predictions = get_predictions_and_save_progress(task, unpredicted_df, taskPath)
            all_predictions.extend(new_predictions)
        
        # Load all rows that we have predictions for
        all_rows = pd.read_csv(os.path.join("dataPool", clean_fileName))
        
        # Merge rows with predictions on row_number
        pred_df = pd.DataFrame(all_predictions)
        pred_df['row_number'] = pred_df['row_number'].astype(int)
        all_rows['row_number'] = all_rows['row_number'].astype(int)
        
        train_df = pd.merge(all_rows, pred_df, on='row_number')
        
        if train_df.empty:
            print("No training data available.")
            break
            
        # Train the model
        model, metrics = model_trainer.train_model(train_df)
        print("Model training complete.")
        
        # Save the trained model file and update path in JSON
        model_path = save_model(taskPath, model, len(all_predictions), taskDetails["taskName"], taskDetails["taskId"])
        print(f"Model saved to '{model_path}' and linked to JSON.")
        
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
                target_rows_count = len(all_predictions) + additional
            else:
                print("Model performance is satisfactory. Retraining not required.")
        except Exception as e:
            print(f"Failed to evaluate metrics via Gemini: {e}. Stopping retraining loop.")
            retrain_needed = False
            
    # Perform final predictions on all rows of the dataset if training succeeded
    if model is not None:
        try:
            print("\nGenerating final student model predictions on all rows in the dataset...")
            # Load all rows from cleaned dataset
            dataset_path = os.path.join("dataPool", clean_fileName)
            df_all = pd.read_csv(dataset_path)
            
            # Predict
            y_pred, confidence = model_trainer.predict_with_confidence(model, df_all['text'])
            
            conf_threshold = eval_res.get("confidence_threshold", 0.8) # Default fallback
            min_trust_threshold = eval_res.get("min_trust_threshold", 0.6) # Default fallback
            
            final_predictions = []
            low_confidence_count = 0
            untrustworthy_count = 0
            
            for i, (index, row) in enumerate(df_all.iterrows()):
                score = float(y_pred[i])
                conf = float(confidence[i])
                is_low = conf < conf_threshold
                is_untrustworthy = conf < min_trust_threshold
                
                if is_low:
                    low_confidence_count += 1
                if is_untrustworthy:
                    untrustworthy_count += 1
                    
                final_predictions.append({
                    "row_number": int(row["row_number"]),
                    "score": score,
                    "confidence": conf,
                    "low_confidence": is_low,
                    "untrustworthy": is_untrustworthy
                })
                
            # Write final predictions and counts to the task JSON file
            with open(taskPath, "r") as f:
                content = json.load(f)
                
            content["final_predictions"] = final_predictions
            content["confidence_threshold"] = conf_threshold
            content["min_trust_threshold"] = min_trust_threshold
            content["low_confidence_predictions_count"] = low_confidence_count
            content["untrustworthy_predictions_count"] = untrustworthy_count
            
            with open(taskPath, "w") as f:
                json.dump(content, f, indent=4)
                
            print(f"\n==================================================")
            print(f"★ AUTOMATED TRAINING PIPELINE COMPLETE ★")
            print(f"==================================================")
            print(f"Total dataset rows predicted: {len(df_all)}")
            print(f"Audit Threshold (Low Confidence): {conf_threshold}")
            print(f"⚠️ LOW CONFIDENCE PREDICTIONS: {low_confidence_count} ⚠️")
            print(f"--------------------------------------------------")
            print(f"Critical Threshold (Untrustworthy): {min_trust_threshold}")
            print(f"❌ UNTRUSTWORTHY PREDICTIONS: {untrustworthy_count} ❌")
            print(f"Full results stored in '{taskPath}'")
            print(f"==================================================")
            
        except Exception as e:
            print(f"Failed to generate final predictions: {e}")
            
    print("Pipeline execution complete.")

if __name__ == "__main__":
    doTask("Which are positive and negative reviews", "temp/IMDB_Dataset.csv")