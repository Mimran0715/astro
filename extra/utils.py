import sqlite3
import pandas as pd

def db_command(path, command, task):
    conn = sqlite3.connect(path)
    c = conn.cursor()
     #print(type(command), type(task))
    if len(task) == 0:
         c.execute(command)
    else:
        c.execute(command, task)
    conn.commit()
    conn.close()
    return c.lastrowid

def db_command_2(path, conn, command, task):
    #conn = sqlite3.connect(path)
    c = conn.cursor()
     #print(type(command), type(task))
    if len(task) == 0:
         c.execute(command)
    else:
        c.execute(command, task)
    conn.commit()
    conn.close()
    return c.lastrowid

def get_tbl_count(db_path, tbl_name):
    conn = sqlite3.connect(db_path)
    entry_count = db_command_2(db_path, conn, "SELECT COUNT(*) FROM " + str(tbl_name) + ";")
    conn.close()

def get_data(db_path, tbl_name, size):

    #print("in function: get_data")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    #entry_count = db_command_2(db_path, conn, "SELECT COUNT(*) FROM " + str(tbl_name) + ";")

    df_1 = pd.read_sql_query("SELECT  * FROM " + str(tbl_name) + " WHERE year >= 1980 AND year <1990 LIMIT " + str(size) +  ";", conn)
    df_2 = pd.read_sql_query("SELECT  * FROM " + str(tbl_name) + " WHERE year >= 1990 AND year <2000 LIMIT " + str(size) +  ";", conn)
    df_3 = pd.read_sql_query("SELECT  * FROM " + str(tbl_name) + " WHERE year >= 2000 AND year <2010 LIMIT " + str(size) +  ";", conn)
    df_4 = pd.read_sql_query("SELECT  * FROM " + str(tbl_name) + " WHERE year >= 2010 AND year <2022 LIMIT " + str(size) +  ";", conn)
    df = pd.concat([df_1, df_2, df_3, df_4], axis=0)

    # print("DF INFO --------->")
    # print(df.info(verbose=False, memory_usage="deep"))
    # print()

    if size == -1:
        df = pd.read_sql_query("SELECT  bibcode, abstract FROM " + str(tbl_name) + ";", conn)
    conn.close()
    return df