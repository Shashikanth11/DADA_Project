from supabase import create_client
import json
import os
from dotenv import load_dotenv

# Load secrets from .env file
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Base path to attack_families folder
BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "attack_families")

if not os.path.exists(BASE_PATH):
    print(f"Error: Folder {BASE_PATH} does not exist.")
    exit()

all_mapped_rows = []

# Loop through all JSON files in the folder
for filename in os.listdir(BASE_PATH):
    if filename.endswith(".json"):
        file_path = os.path.join(BASE_PATH, filename)

        try:
            with open(file_path, "r") as f:
                rows = json.load(f)
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            continue
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {file_path}. Skipping file.")
            continue

        # Derive attack_family from filename (remove extension)
        attack_family = os.path.splitext(filename)[0]

        # Map JSON keys to match Supabase table columns
        for r in rows:
            all_mapped_rows.append({
                "attack_family": attack_family,
                "attack_name": r.get("attack_name", ""),
                "attack_prompt": r.get("attack_prompt", ""),
                "usecase": r.get("usecase", ""),
            })

if not all_mapped_rows:
    print("No data to upload.")
    exit()

# Insert rows into Supabase
response = supabase.table("prompt_injection_attacks").insert(all_mapped_rows).execute()

print("Attack data uploaded successfully.")
print(f"Inserted {len(all_mapped_rows)} records.")
