import sqlite3

conn = sqlite3.connect('/home/maleeha/research/data/research.db')
#conn = sqlite3.connect('/Users/Mal/Desktop/sqlite/research.db')
c = conn.cursor()
c.execute("CREATE TABLE astro_papers_1980_2021 \
(bibcode TEXT, title TEXT, year INTEGER, author TEXT, abstract TEXT, citation_count INTEGER,\
    file_path TEXT, downloaded_pdf INTEGER, ran_sentiment INTEGER, sentiment REAL, \
        paper_text TEXT, word_vector BLOB);")
c.execute("CREATE TABLE astro_papers_1991_2000 \
(bibcode TEXT, title TEXT, year INTEGER, author TEXT, abstract TEXT, citation_count INTEGER,\
    file_path TEXT, downloaded_pdf INTEGER, ran_sentiment INTEGER, sentiment REAL, \
        paper_text TEXT, word_vector BLOB);")
c.execute("CREATE TABLE astro_papers_2001_2010 \
(bibcode TEXT, title TEXT, year INTEGER, author TEXT, abstract TEXT, citation_count INTEGER,\
    file_path TEXT, downloaded_pdf INTEGER, ran_sentiment INTEGER, sentiment REAL, \
        paper_text TEXT, word_vector BLOB);")
c.execute("CREATE TABLE astro_papers_2011_2021 \
(bibcode TEXT, title TEXT, year INTEGER, author TEXT, abstract TEXT, citation_count INTEGER,\
    file_path TEXT, downloaded_pdf INTEGER, ran_sentiment INTEGER, sentiment REAL, \
        paper_text TEXT, word_vector BLOB);")
conn.commit()
conn.close()

