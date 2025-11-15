from supabase import create_client
import json
import os
from dotenv import load_dotenv

# Load secrets from .env file
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Base path relative to project root
BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "benchmark_data")

# Map usecases to their json files
json_files = {
    "e-commerce": os.path.join(BASE_PATH, "e-commerce.json"),
    "banking": os.path.join(BASE_PATH, "banking.json"),
    "academic": os.path.join(BASE_PATH, "academic.json"),
    "insurance": os.path.join(BASE_PATH, "insurance.json"),
}

# Loop through each file and insert into Supabase
for usecase, filepath in json_files.items():
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue

    with open(filepath, "r") as f:
        data = json.load(f)

    response = supabase.table("business_data").insert({
        "usecase": usecase,
        "data": data  # jsonb column
    }).execute()

    #print(f"Uploaded {usecase}: {response}")

print("All data inserted.")