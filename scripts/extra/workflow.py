
import sys
from urllib.parse import urlencode, quote_plus
import requests
import os
from urllib.request import urlretrieve
import sqlite3
import pdfplumber as p
import time
import sys
import arxiv
import json

def db_command(path, command, task):
     conn = sqlite3.connect(path)
     c = conn.cursor()
     #print(type(command), type(task))
     c.execute(command, task)
     conn.commit()
     conn.close()
     return c.lastrowid

def main():
    db_path = '/Users/Mal/Documents/research.db'
    tbl_path = 'astro_papers_t1'
    db_command(db_path, \
        "INSERT INTO " + str(tbl_path) + "(bibcode, \
        alternate_bibcode , \
        title  , \
        date , \
        year , \
        doctype , \
        eid , \
        recid , \
        esources , \
        property, \
        citation , \
        read_count , \
        author,\
        abstract, \
        citation_count, \
        identifier, \
        file_path, \
        downloaded_pdf,\
        ran_sentiment, \
        sentiment, \
        paper_text) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);", \
        (paper['bibcode'], paper['title'][0], int(paper['year']), \
        ' '.join(paper['author']), paper['abstract'], int(paper['citation_count']), \
            file_name, 1, 0, 0.0, text, None))
    pass

if __name__ == "__main__":
    main()