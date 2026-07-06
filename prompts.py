TASK_CONFIRMATION = """System Role: You are a JSON-only data matcher and dataset Cleaner.
Task-1: Compare the target task with the list of tasks in the rows.
Task-2: Give me a command to clean the dataset with the columns that it has so that i am only left with (row_number,text) ignore Previous attempst. give me the cleanest command
Input: 
- dataSet_info : {}
- target: {}
- rows: {}



Logic:
if there are no rows or If no semantically similar task is found, output {{"same": 0,"taskId":<next_taskId(natural)>, "taskName" : <proper_max_5_word_name>,"datasetClean" : <a dataset cleanning command using pandas datafram named 'df'>}}.
If the target task has the same meaning, intent, or is highly related to a task in the rows, output {{"same": 1, "taskId": <id>}}.

Constraints:
Return only the JSON object.
No markdown formatting, no explanations, no text."""

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
1. If the R-squared (r2) score on unseen data (unseen_data_metrics) is less than 0.8, the model performance is considered poor or underfitting. It needs more training data.
2. If more data is needed, set "retrain" to true, and suggest "additional_rows" to fetch (e.g., 50, 100, or 200 depending on how poor the metrics are).
3. If the R-squared score on unseen data is 0.8 or higher, or if we have reached a limit of 1000 total rows, set "retrain" to false.

Output JSON format:
{{
  "retrain": boolean,
  "additional_rows": number,
  "reason": "string explaining why"
}}

Respond only with the valid JSON object. No explanation, no markdown blocks."""