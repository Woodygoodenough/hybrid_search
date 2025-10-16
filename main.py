# %% imports
import pandas as pd
from settings import USECOLS, CHUNK_SIZE, OUT_CSV
from dbManagement import DbManagement


def main():
    db_management = DbManagement()
    db_management.create_db()
    reader = pd.read_csv(OUT_CSV, usecols=USECOLS, chunksize=CHUNK_SIZE)
    db_management.load_db(reader)

if __name__ == "__main__":
    main()
