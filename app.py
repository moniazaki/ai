from fastapi import FastAPI, HTTPException
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load the LLaMA model and tokenizer from the local directory
model_directory ="D:/Desktop/RATP/llama-models/models/llama3_1" 
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(model_directory)

# Load and preprocess the CSV file
def load_and_preprocess_csv(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Clean the data (example: drop rows with missing values)
    df.dropna(inplace=True)
    
    # Convert all text to lowercase for uniformity
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)
    
    # Additional preprocessing steps (e.g., removing special characters)
    df.replace(r'[^\w\s]', '', regex=True, inplace=True)
    
    return df

# Load CSV data
data = load_and_preprocess_csv('./sample.csv')  # Adjust the path to your CSV file

def query_model(user_query, df):
    # Extract relevant data based on the user's query
    relevant_data = df[df.apply(lambda row: row.astype(str).str.contains(user_query).any(), axis=1)]
    
    # If no relevant data is found, return an appropriate message
    if relevant_data.empty:
        return "No relevant data found for your query."
    
    # Construct the context for the LLaMA model
    context = f"User Query: {user_query}\nRelevant Data:\n{relevant_data.to_string(index=False)}"
    
    # Tokenize and generate a response from the model
    inputs = tokenizer(context, return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# API endpoint to handle queries
@app.post("/query")
def handle_query(query: str):
    try:
        response = query_model(query.lower(), data)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

model.save_pretrained("./Meta-Llama3.1-8B")
tokenizer.save_pretrained("./Meta-Llama3.1-8B")