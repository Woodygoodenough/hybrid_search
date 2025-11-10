python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt


# 1. Fetch Wikipedia data (150k articles)
python datafetch.py

# 2. Add sequential IDs to CSV
python add_item_id_to_csv.py wikipedia_sample_150k.csv

# 3. Create metadata database
python dbManagement.py

# 4. Build FAISS vector index
python create_faiss_from_csv.py wikipedia_sample_150k_with_ids.csv


pytest -v test_timing.py

