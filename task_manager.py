import os
import json
import getData
import prompts
import getGeminiResponse as gemini

def get_task_details(task, taskFile):
    """Confirm the task and clean instructions via Gemini."""
    taskManaged = os.listdir("tasks")
    dataSetInfo = getData.onlyHeaders(taskFile)
    
    prompt = prompts.TASK_CONFIRMATION.format(task, str(taskManaged), str(dataSetInfo))
    taskDetails = gemini.getResponse(prompt)
    return taskDetails

def initialize_task_file(taskDetails, taskFile):
    """Clean the data pool file and prepare the task JSON configuration file."""
    fileName = str(taskDetails["taskId"]) + "_" + taskDetails["taskName"].replace(" ", "-") + ".json"
    clean_fileName = getData.cleanDataFrame(taskFile, taskDetails['datasetClean'], fileName)
    
    taskPath = os.path.join("tasks", fileName)
    content = {
        "taskId": taskDetails["taskId"],
        "taskName": taskDetails["taskName"],
        "fileName": clean_fileName,
        "rows_predicted": 0,
        "best_model_metricts": None,
        "rows_used": []
    }
    
    with open(taskPath, "w") as f:
        json.dump(content, f, indent=4)
        
    return taskPath, clean_fileName

def update_task_progress(taskPath, new_predictions):
    """
    Read the task JSON file, append new predictions, increment rows_predicted,
    and save the file back.
    """
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
