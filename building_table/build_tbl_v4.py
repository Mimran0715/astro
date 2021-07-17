import sqlite3
import sys

#def construct_tbl(path:str, fields:dict, table_name:str):
def construct_tbl(path:str, table_name:str):
    conn = sqlite3.connect(path)
    #conn = sqlite3.connect('/home/maleeha/research/data/research.db')
    #conn = sqlite3.connect('/Users/Mal/Desktop/sqlite/research.db')
    c = conn.cursor()
    c.execute("CREATE TABLE " + table_name +  \
        "(bibcode TEXT, \
        alternate_bibcode TEXT, \
        title TEXT, \
        date TEXT, \
        year TEXT, \
        doctype TEXT, \
        eid TEXT, \
        recid TEXT , \
        esources TEXT, \
        property TEXT, \
        citation TEXT, \
        read_count TEXT, \
        author TEXT,\
        abstract TEXT, \
        citation_count TEXT, \
        identifier TEXT, \
        file_path TEXT, \
        downloaded_pdf INTEGER,\
        ran_sentiment INTEGER, \
        sentiment REAL, \
        paper_text TEXT, \
        abs_text TEXT, \
        paper_proc_text);")
    conn.commit()
    conn.close()

def main():
    # enter like following: python3.7 build_db_2.py path tblname
    db_path = sys.argv[1]
    tbl_name = sys.argv[2]
    field_list = []
    construct_tbl(db_path, tbl_name)

if __name__ == '__main__':
    main()