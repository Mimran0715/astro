import sys
from urllib.parse import urlencode, quote_plus
import requests
import os
from urllib.request import urlretrieve
import sqlite3
#import pdfplumber as p
import time
import sys
#import arxiv
import json
#from os import listdir, mkdir
#import text_process as t
#import sqlite_db

from utils import db_command
request_count = 0

API_TOKEN = "4EqqwYzTgJYNypXmhwVZNA4UiDytgRdqbcxURdD9"

def build_url(start: int, params:dict, fields:list, rows:int, sort:str='bibcode+desc') -> str:
    base_url = "https://api.adsabs.harvard.edu/v1/search/query?q="
    query = "year:" + params['start_year'] + "-" + params['end_year']
    encoded_query = quote_plus(query)
    num_fields = len(fields)
    field_str = "&fl="
    for i in range(num_fields-1):
        field_str += fields[i] + ','
    field_str += fields[num_fields-1]
    url = base_url + encoded_query + "&fq=database:astronomy" + "&rows=" + str(rows) + "&start=" + str(start) + field_str
    print("URL: ", url)
    print()
    #url = base_url + encoded_query + "&fq=database:astronomy,start=" + str(start) + "&" + "fl=id,links_data,\
    # bibcode,doi,arxiv_class,title,year,author,abstract,citation_count,identifier&" + "rows=2000"
    #url = base_url + encoded_query + "&fq=database:astronomy&" + "rows=2000"
    return url

def obtain_ads_data(start: int, params: dict, db: int, f_path:str, db_path:str, table_name:str, fields:list, rows:int) -> None:
    global request_count
    ads_url = build_url(start, params, fields, rows)
    ads_response= requests.get(ads_url,headers={'Authorization': 'Bearer ' + API_TOKEN}).json()

    #print(ads_response['response'])
    
    #with open('data.txt', 'w') as of:
    #    json.dump(ads_response, of)

    #folder_count = 1
    missing_count = 0
    total_count = 0
    
    #paper_count = 0
    s = start
    
    while s <= ads_response['response']['numFound']:
    #while ads_response['response']['numFound'] != 0:
        papers = ads_response['response']['docs']
        for paper in papers:
            #print("id", paper['id'])
            #return
            # #attempting to download pdfs if available
            # folder_path = f_path + str(folder_count) 
            # if(os.path.exists(folder_path)==0):
            #     os.mkdir(folder_path)

            # if paper_count == 5000: #make a new folder
            #     folder_count +=1
            #     folder_path = f_path + str(folder_count)
            #     if(os.path.exists(folder_path)==0):
            #         os.mkdir(folder_path)   
            #     paper_count = 0 

            # file_name = folder_path + '/' + paper['bibcode'] + ".pdf"

            # try:
            #     if 'EPRINT' in paper['property']:
            #         eprint_url = "https://ui.adsabs.harvard.edu/link_gateway/" + paper['bibcode'] + "/EPRINT_PDF" 
            #         urlretrieve(url, file_name)                    
            # except KeyError:
            #     print("No Property Key")
            #     try:
            #         if 'EPRINT_PDF' in paper['esources']:
            #             eprint_url = "https://ui.adsabs.harvard.edu/link_gateway/" + paper['bibcode'] + "/EPRINT_PDF" 
            #             urlretrieve(url, file_name) 
            #     except KeyError:
            #         print("No Esources Key")
            #         try: 
            #             if paper['doctype'] == 'eprint':
            #                 eprint_url = "https://ui.adsabs.harvard.edu/link_gateway/" + paper['bibcode'] + "/EPRINT_PDF" 
            #                 urlretrieve(url, file_name)
            #         except KeyError:
            #             print("No doctype key")
            #break

            #getting fields to place into sqlite table
            val_dict = {}
            for field in fields:
                try:
                    #print("Field Name: " + field)
                    #print("Type of Return Data: ", type(paper[field]))
                    #print()
                    val = None
                    if type(paper[field]) == list:
                        val = ' '.join(paper[field])
                    elif type(paper[field]) == int:
                        val = str(paper[field])
                    else:
                        val = paper[field]
                    val_dict[field] = val
                except KeyError:
                    #print("field does not exist in current return")
                    #print()
                    val_dict[field] = None

            for k, v in val_dict.items():
                print(k, v)

            #values = list(val_dict.values()) 
            #extra_fields = fields + ['file_path','downloaded_pdf', 'ran_sentiment', 'sentiment', 'paper_text', 'abs_text', 'paper_proc_text']
            
            #p_fields = list(paper.keys())
            #print("field len: ", len(fields), "pfield len:", len(p_fields))

            # vals = " VALUES ("
            # for i in range(len(fields)-1):
            #     vals += "?,"
            # vals += '?);'
        
            # cmd_str =  "INSERT INTO " + table_name + " " + str(tuple(extra_fields)) + vals
            # print(cmd_str)
            # print(len(values))

            try:
                cmd_str = "INSERT INTO " + str(table_name) + "(id, bibcode,title,year,author,abstract,\
                    citation_count) VALUES (?, ?, ?, ?, ?, ? ,?); "
                db_command(db_path, cmd_str, (paper['id'], paper['bibcode'], paper['title'][0],\
                     int(paper['year']), ' '.join(paper['author']), paper['abstract'], \
                         int(paper['citation_count']) ) )
            except:
                print("Issue with inserting entry into db...Fields available: ", list(paper.keys()))
                print()
                missing_count+=1

            total_count +=1
            #paper_count +=1
        request_count +=1 
        print("request {r} done...".format(r=request_count))
        #return
        if(request_count == 100):
            print("100 big query requests done... ending download for today")
            #print("Current start value: ", start)
            print("Current start value: ", s)
            print("new_url", new_url)
            print("ads_url", ads_url)
            return

        time.sleep(2)
        #start += 2000
        s += 2000
        new_url = build_url(s, params, fields, rows)
        ads_response= requests.get(new_url,headers={'Authorization': 'Bearer ' + API_TOKEN}).json()
        #print("End start val", ads_response['response']['start'])
    #print("Papers not entered into db: ", missing_count)
    print("Missing number of papers: ", missing_count)
    print("Papers gotten: ", total_count - missing_count)

def main():
    db_path = '/Users/Mal/Documents/research.db'
    folder_path = '/Users/Mal/Documents/pdfs/'
    #db_no = 0

    run_loc = int(sys.argv[1]) # local == 0, clotho == 1
    table_name = sys.argv[2]
    num_decades = sys.argv[3] # 0 - 1980s only, etc...

    if run_loc == 1:
        db_path = '/home/maleeha/research/research.db'
        folder_path = '/home/maleeha/research/pdfs/'
        db_no = 1

    elif run_loc == 2:
        db_path = '/Users/Mal/Desktop/t.db'
        folder_path = '/Users/Mal/Desktop/pdfs/'

    if(os.path.exists(folder_path)==0):
        os.mkdir(folder_path)
    
    fields = ['id', 'bibcode','title','year','author','abstract','citation_count']

    '''fields_long = ['bibcode', 'alternate_bibcode', 'title', 'date' , 'year', 'doctype', \
        'eid', 'recid', 'esources', 'property', 'citation', 'read_count', 'author', \
            'abstract', 'citation_count', 'identifier']'''

    start = int(input("enter start value"))
    rows = 2000

    year_list = [('1980', '1990'), ('1991', '2000'), ('2001', '2010'), ('2011', '2021')]

    for i in range(int(num_decades)):
        print("starting download of: ", year_list[i][0], " - ", year_list[i][1])
        curr_params = {'start_year':year_list[i][0], 'end_year':year_list[i][1]}
        obtain_ads_data(start, curr_params, db_no, folder_path, db_path, table_name, fields, rows)
        print("Done with: ", year_list[i][0], " - ", year_list[i][1])
  
if __name__ == '__main__':
    main()

# obtain_ads_data_code:
# try:py
            #     print('types of docs:')
            #     print()
            #     print('doctype: ', paper['doctype'])
            #     print('esources: ', paper['esources'])
            #     print('property: ', paper['property'])
            #     print()
            #     print('types of ids:')
            #     print()
            #     print('eid: ', paper['eid'])
            #     print('recid: ', paper['recid'])
            #     print('bibcode: ', paper['bibcode'])
            #     print('alternate_bibcode: ', paper['alternate_bibcode'])
            #     print('identifier: ', paper['identifer'])
            #     print()
                # if 'eprint' in paper['doctype']:
                #     print('got the eprint')
                # else:
                #     print(paper['doctype'])
                    
            # except KeyError:
            #     print("no doctype")
            # try:
            #     if 'ADS_PDF' in paper['esources']:
            #         eprint_url = "https://ui.adsabs.harvard.edu/link_gateway/" + paper['bibcode'] + "/EPRINT_PDF" 
            #         pub_url = "https://ui.adsabs.harvard.edu/link_gateway/" + paper['bibcode'] + "/PUB_PDF" 
            #         r = requests.get(eprint_url)
            #         print("eprint url", r.url)
            #         print("eprint history", r.history)

            #         rp = requests.get(pub_url)
            #         print("pub url", rp.url)
            #         print("pub history", rp.history)

            # except KeyError:
            #     print("key error pdf")