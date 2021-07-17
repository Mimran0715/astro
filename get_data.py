import sqlite3
import pandas as pd
import sys

def main():
    run_loc = sys.argv[1]  # local == 0, clotho == 1

    if run_loc == 0:
        db_path = '/Users/Mal/Documents/research.db'
    else:
        db_path = '/home/maleeha/research/research.db'
        
    tbl_name = "astro_papers_t3"
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM " + str(tbl_name) + ";", conn)
    conn.close()

    df.to_csv(path_or_buf="./data/data.csv", index=False)

if __name__ == "__main__":
    main()