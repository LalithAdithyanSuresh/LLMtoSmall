TASK_CONFIRMATION = """System Role: You are a JSON-only data matcher and dataset Cleaner.
Task-1: Compare the target task with the list of tasks in the rows.
Task-2: Give me a command to clean the dataset with the columns that it has so that i am only left with (row_number,text) ignore Previous attempst. give me the cleanest command
Input: 
- dataSet_info : {}
- target: {}
- rows: {}



Logic:
if there are no rows or If no semantically similar task is found, output {{"same": 0,"taskId":<next_taskId(natural)>, "taskName" : <proper_max_5_word_name>,"datasetClean" : <a dataset cleanning command using pandas dataframe named 'df'>}}.
If the target task has the same meaning, intent, or is highly related to a task in the rows, output {{"same": 1, "taskId": <id>}}.

Dataset Cleaning Rules (MUST FOLLOW):
1. The datasetClean command must result in a pandas DataFrame named 'df' with exactly two columns: 'row_number' (integer) and 'text' (string).
2. You MUST rename the column containing the main text content to 'text' (e.g., if the column is named 'review', rename it to 'text' using `.rename(columns={{'review': 'text'}})`).
3. Create 'row_number' from the index (e.g., `.reset_index().rename(columns={{'index': 'row_number'}})`).
4. Example: "df = df.dropna(subset=[['review']]).reset_index().rename(columns={{'index': 'row_number', 'review': 'text'}})[['row_number', 'text']]"

Constraints:
Return only the JSON object.
No markdown formatting, no explanations, no text.
Do not wrap in python code blocks. Output raw JSON only."""

GET_SCORE_PREDICTIONS = """Act as an expert data analyst and machine learning engineer. Your task is to perform a regression analysis on the provided dataset to classify the following goal: {{ $('On form submission').item.json['Task to do'] }}

For each item in the input list, perform the following:

Analyze the text content for the specific classification task defined above.

Assign a continuous numerical score between 0.0 and 1.0, where 0.0 indicates a complete mismatch or negative result, and 1.0 indicates a perfect match or positive result.

Provide a concise, professional explanation for the score assigned.

You must return the output as a strict, valid JSON array containing objects with the following schema:
[
{{
"row_number": number,
"score": float,
"reason": "string"
}}
]

Ensure the output contains no conversational filler, no markdown code block formatting (unless strictly required by the system), and adheres strictly to the JSON structure provided. Here is the dataset to process (make sure to do for all rows):
{}"""

EVALUATE_MODEL_METRICS = """System Role: You are an expert machine learning advisor.
Your task is to analyze the training and validation metrics of a trained regression model and decide whether it needs retraining with more data.

Input:
- Model metrics: {}
- Number of rows already trained: {}

Rules:
1. Dynamically evaluate model metrics. The ultimate goal is to use this student model to run predictions on the remaining dataset (5728 rows total) to save LLM API token costs.
2. Balance performance vs. cost: If the R-squared (r2) score on unseen data (unseen_data_metrics) is showing diminishing returns (e.g. barely improving from 0.78 to 0.79 with more data), or if it is already reasonably high (e.g. ~0.75 or above) and sufficient for the task, set "retrain" to false to save on expensive Gemini labeling runs.
3. Only request retraining ("retrain": true) if the performance is clearly inadequate and showing strong potential to improve significantly with more targeted training rows.
4. Recommend a "confidence_threshold" (float, e.g., 0.75-0.85) representing the minimum acceptable prediction confidence based on the validation uncertainty metrics.
5. Recommend a "min_trust_threshold" (float, e.g., 0.60-0.70) representing the absolute lower limit of confidence below which the model's prediction is completely untrustworthy.

Output JSON format:
{{
  "retrain": boolean,
  "additional_rows": number,
  "confidence_threshold": float,
  "min_trust_threshold": float,
  "reason": "string explaining why"
}}

Respond only with the valid JSON object. No explanation, no markdown blocks."""