import pandas as pd
import sqlite3
import sys

def main():
    db_path = sys.argv[0]
    tbl_name = sys.argv[1]
    conn = sqlite3.connect(db_path)
    for chunk in pd.read_sql("SELECT * FROM " + str(tbl_name) + ";", conn, 'bibcode', 2000):
        pass


    pass

if __name__ == "__main__":
    main()