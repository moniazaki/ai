from fastapi import FastAPI, HTTPException
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load the LLaMA model and tokenizer
model_name = "LLaMA"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load and preprocess multiple CSV files
def load_and_preprocess_csv(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    df = df.applymap(lambda s: s.lower() if isinstance(s, str) else s)
    df.replace(r'[^\w\s]', '', regex=True, inplace=True)
    return df

# Dictionary to store the DataFrames with filenames as keys
csv_data = {}

# List of file paths to load
csv_files = [
    "file1.csv",
    "file2.csv",
    "file3.csv"
]

# Load all CSV files at startup
for file_path in csv_files:
    csv_data[file_path] = load_and_preprocess_csv(file_path)

# Function to handle user query and model interaction
def query_model(user_query, df):
    relevant_data = df[df.apply(lambda row: row.astype(str).str.contains(user_query).any(), axis=1)]
    if relevant_data.empty:
        return "No relevant data found for your query."
    
    context = f"User Query: {user_query}\nRelevant Data:\n{relevant_data.to_string(index=False)}"
    inputs = tokenizer(context, return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# API endpoint to handle queries
@app.post("/query")
async def handle_query(query: str, file_name: str):
    if file_name not in csv_data:
        raise HTTPException(status_code=400, detail="CSV file not found.")
    
    df = csv_data[file_name]
    
    try:
        response = query_model(query.lower(), df)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
