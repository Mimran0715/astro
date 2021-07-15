import sqlite3

last_row_id = 0

def main():
    db_path = "/Users/Mal/Documents/research.db"
    conn = sqlite3.connection(db_path)
    if last_row_id < conn.lastrowid:
        
        # this means more rows have been added, so now we have to preprocess the data in those rows
        # using the last_row_id 
        last_row_id = conn.lastrowid
        pass

if __name__ == "__main__":
    main()