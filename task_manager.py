import os
import json
import getData
import prompts
import getGeminiResponse as gemini
import threading

file_lock = threading.Lock()

def get_task_details(task, taskFile):
    """Confirm the task and clean instructions via Gemini."""
    taskManaged = os.listdir("tasks")
    dataSetInfo = getData.onlyHeaders(taskFile)
    
    prompt = prompts.TASK_CONFIRMATION.format(str(dataSetInfo), task, str(taskManaged))
    taskDetails = gemini.getResponse(prompt)
    return taskDetails

def initialize_task_file(taskDetails, taskFile):
    """Clean the data pool file and prepare the task JSON configuration file."""
    fileName = str(taskDetails["taskId"]) + "_" + taskDetails["taskName"].replace(" ", "-") + ".json"
    taskPath = os.path.join("tasks", fileName)
    
    # If the file already exists, load its content and DO NOT overwrite it
    with file_lock:
        if os.path.exists(taskPath):
            try:
                with open(taskPath, "r") as f:
                    existing_content = json.load(f)
                # If it has some predicted rows, reuse them
                if existing_content.get("rows_used"):
                    print(f"Task file '{taskPath}' already exists with {len(existing_content['rows_used'])} predictions. Reusing it.")
                    return taskPath, existing_content["fileName"], existing_content["rows_used"]
            except Exception as e:
                print(f"Error reading existing task file: {e}. Overwriting.")

    clean_fileName = getData.cleanDataFrame(taskFile, taskDetails['datasetClean'], fileName)
    
    content = {
        "taskId": taskDetails["taskId"],
        "taskName": taskDetails["taskName"],
        "fileName": clean_fileName,
        "rows_predicted": 0,
        "best_model_metricts": None,
        "rows_used": []
    }
    
    with file_lock:
        with open(taskPath, "w") as f:
            json.dump(content, f, indent=4)
        
    return taskPath, clean_fileName, []

def update_task_progress(taskPath, new_predictions):
    """
    Read the task JSON file, append new predictions, increment rows_predicted,
    and save the file back (thread-safe).
    """
    with file_lock:
        if not os.path.exists(taskPath):
            return
            
        with open(taskPath, "r") as f:
            content = json.load(f)
            
        content["rows_used"].extend(new_predictions)
        content["rows_predicted"] = len(content["rows_used"])
        
        with open(taskPath, "w") as f:
            json.dump(content, f, indent=4)

def update_task_metrics(taskPath, metrics):
    """
    Update the final model metrics in the task JSON file.
    """
    if not os.path.exists(taskPath):
        return
        
    with open(taskPath, "r") as f:
        content = json.load(f)
        
    content["best_model_metricts"] = metrics
    
    with open(taskPath, "w") as f:
        json.dump(content, f, indent=4)

def load_existing_task(taskId):
    """
    Search in the tasks directory for a file starting with '<taskId>_'
    and load its JSON content. Returns (taskPath, content) or (None, None).
    """
    task_dir = "tasks"
    if not os.path.exists(task_dir):
        return None, None
        
    prefix = f"{taskId}_"
    for filename in os.listdir(task_dir):
        if filename.startswith(prefix) and filename.endswith(".json"):
            taskPath = os.path.join(task_dir, filename)
            try:
                with open(taskPath, "r") as f:
                    content = json.load(f)
                return taskPath, content
            except Exception as e:
                print(f"Error loading existing task file {filename}: {e}")
    return None, None

def save_model(taskPath, model, row_count, taskName, taskId):
    """
    Save the trained scikit-learn model to models/ and reference it in the task JSON.
    """
    import joblib
    os.makedirs("models", exist_ok=True)
    model_filename = f"{taskId}_{taskName.replace(' ', '-')}_{row_count}.joblib"
    model_path = os.path.join("models", model_filename)
    joblib.dump(model, model_path)
    
    with open(taskPath, "r") as f:
        content = json.load(f)
    content["model_path"] = model_path
    with open(taskPath, "w") as f:
        json.dump(content, f, indent=4)
        
    return model_path

