import sqlite3
import sys
def construct_tbl(path:str,table_name:str):
    conn = sqlite3.connect(path)
    #conn = sqlite3.connect('/home/maleeha/research/data/research.db')
    #conn = sqlite3.connect('/Users/Mal/Desktop/sqlite/research.db')
    c = conn.cursor()
    c.execute("CREATE TABLE " + table_name +  \
        "(id INTEGER, \
        bibcode TEXT, \
        title TEXT, \
        year INTEGER, \
        author TEXT,\
        abstract TEXT, \
        citation_count INTEGER \
        );")
        
    conn.commit()
    conn.close()

def main():
    # enter like following: python3.7 build_db_2.py path
    db_path = sys.argv[1]
    tbl_name = sys.argv[2]
    construct_tbl(db_path, tbl_name)

if __name__ == '__main__':
    main()