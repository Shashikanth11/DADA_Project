from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

table_name = "prompt_injection_attacks"

# Step 1: Fetch all rows
all_rows = supabase.table(table_name).select("*").execute().data

# Step 2: Keep only unique rows based on (usecase, attack_prompt)
seen = set()
unique_rows = []
for row in all_rows:
    key = (row["usecase"], row["attack_prompt"])
    if key not in seen:
        seen.add(key)
        unique_rows.append(row)
print("Attack Table")
print(f"[INFO] Found {len(all_rows)} rows, {len(unique_rows)} unique rows")

# Step 3: Delete all rows
supabase.table(table_name).delete().neq("id", 0).execute()  # delete everything

# Step 4: Re-insert unique rows
if unique_rows:
    supabase.table(table_name).insert(unique_rows).execute()
    print(f"[INFO] Re-inserted {len(unique_rows)} unique rows")

table_name = "business_data"

# Fetch all rows
all_rows = supabase.table(table_name).select("*").execute().data

# Keep only unique rows based on (usecase, data)
seen = set()
unique_rows = []
for row in all_rows:
    key = (row["usecase"], str(row["data"]))  # convert data JSON to string for comparison
    if key not in seen:
        seen.add(key)
        unique_rows.append(row)
print("Business Data Table")
print(f"[INFO] Found {len(all_rows)} rows, {len(unique_rows)} unique rows")

# Delete all rows
supabase.table(table_name).delete().neq("id", 0).execute()

# Re-insert unique rows
if unique_rows:
    supabase.table(table_name).insert(unique_rows).execute()
    print(f"[INFO] Re-inserted {len(unique_rows)} unique rows")
