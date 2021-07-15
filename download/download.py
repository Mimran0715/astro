import sys
from urllib.parse import urlencode, quote_plus
import requests
from os import listdir, mkdir
from urllib.request import urlretrieve
import sqlite3
#import text_process as t
#import sqlite_db
import time

API_TOKEN = "4EqqwYzTgJYNypXmhwVZNA4UiDytgRdqbcxURdD9"

#Building the url to access the NASA ADS API with a year range , ex: year: 1999-2000
def build_url(params:dict) -> str:
    # params is currently : {start_year:value, end_year:value}
    base_url = "https://api.adsabs.harvard.edu/v1/search/query?q="
    query = "year:" + params['start_year'] + "-" + params['end_year']
    encoded_query = quote_plus(query)
    url = base_url + encoded_query + "&fq=database:astronomy&" + "fl=bibcode,title,year,author,abstract,citation_count&" + "rows=2000"
    return url

# Obtaining the ADS response and extracting the pdf url (related to Arxiv) and retrieving the pdf
def obtain_ads_data(url:str, db: int) -> None:
    ads_response= requests.get(url,headers={'Authorization': 'Bearer ' + API_TOKEN}).json()
    #print(type(ads_response['response']['numFound']))
    #paper_count = 0
    folder_count = 1
    #print(ads_response)
    missing_count = 0
    total_count = 0
    request_count = 0
    while ads_response['response']['numFound'] != 0:
        papers = ads_response['response']['docs']
        for paper in papers:
            total_count +=1
            #folder_path = '/Users/Mal/Desktop/research/f' + str(folder_count)
            #if paper_count == 5000:
                # make a new folder
                # folder_count +=1
                # folder_path = '/Users/Mal/Desktop/research/f' + str(folder_count)
                # mkdir(folder_path)   
            #    paper_count = 0
            #pdf_url = "https://ui.adsabs.harvard.edu/link_gateway/" + paper['bibcode'] + "/EPRINT_PDF" 
            #folder_path = '/Users/Mal/Desktop/research/' # change file string based on how many files in folder, < 50,000
            folder_path = '/home/maleeha/research/' # change file string based on how many files in folder, < 50,000
            #urlretrieve(pdf_url, folder_path + paper['title'][0] + ".pdf")
            # input data into the database 
            file_string = folder_path + paper['title'][0] + ".pdf"
            #text = t.obtain_pdf_text(file_string)
            text = ""
            #print(paper['author'])
            try:
                if db == 0:
                    db_command('''INSERT INTO astro_papers_1980_1990 (bibcode,title,year,author,abstract,citation_count,file_path,downloaded_pdf, ran_sentiment,\
                        sentiment, paper_text, word_vector) VALUES (?,?,?,?,?,?,?,?,?,?,?,?);''', (paper['bibcode'], paper['title'][0], int(paper['year']), \
                            ' '.join(paper['author']), paper['abstract'], int(paper['citation_count']), file_string, 1, 0, 0.0, text, None))
            
                    #print('Inserted pdf') # dont know if it should be None above for blob
                elif db == 1:
                    db_command('''INSERT INTO astro_papers_1991_2000 (bibcode,title,year,author,abstract,citation_count,file_path,downloaded_pdf, ran_sentiment,\
                        sentiment, paper_text, word_vector) VALUES (?,?,?,?,?,?,?,?,?,?,?,?);''', (paper['bibcode'], paper['title'][0], int(paper['year']), \
                            ' '.join(paper['author']), paper['abstract'], int(paper['citation_count']), file_string, 1, 0, 0.0, text, None))
            
                    #print('Inserted pdf') # dont know if it should be None above for blob
                elif db == 2:
                    db_command('''INSERT INTO astro_papers_2001_2010 (bibcode,title,year,author,abstract,citation_count,file_path,downloaded_pdf, ran_sentiment,\
                        sentiment, paper_text, word_vector) VALUES (?,?,?,?,?,?,?,?,?,?,?,?);''', (paper['bibcode'], paper['title'][0], int(paper['year']), \
                            ' '.join(paper['author']), paper['abstract'], int(paper['citation_count']), file_string, 1, 0, 0.0, text, None))
            
                    #print('Inserted pdf') # dont know if it should be None above for blob
                elif db == 3:
                    db_command('''INSERT INTO astro_papers_2011_2021 (bibcode,title,year,author,abstract,citation_count,file_path,downloaded_pdf, ran_sentiment,\
                        sentiment, paper_text, word_vector) VALUES (?,?,?,?,?,?,?,?,?,?,?,?);''', (paper['bibcode'], paper['title'][0], int(paper['year']), \
                            ' '.join(paper['author']), paper['abstract'], int(paper['citation_count']), file_string, 1, 0, 0.0, text, None))
        
            except KeyError:
                missing_count +=1
                #print("missing a value")
            #paper_count += 1
        request_count +=1 
        print("request {r} done...".format(r=request_count))
        if(request_count == 50):
            print("50 requests done, now moving on to next year bracket")
            break
        time.sleep(2)
        ads_response= requests.get(url,headers={'Authorization': 'Bearer ' + API_TOKEN}).json()
    print("Missing number of papers bc of KeyError: ", missing_count)
    print("Papers gotten: ", total_count - missing_count)

# Helper function to update database
def db_command(command, task):
     #conn = sqlite3.connect('/home/maleeha/research/research.db')
     conn = sqlite3.connect('/home/maleeha/research/data/research.db')
     #conn = sqlite3.connect('/Users/Mal/Desktop/sqlite/research.db')
     c = conn.cursor()
     #print(type(command), type(task))
     c.execute(command, task)
     conn.commit()
     conn.close()
     return c.lastrowid

def main():
    # creating the database table --> doing in another file so I only have to do it once
    # executing request to API 
    #for i in range(1980, 2021):
    
    # 0 == 1980 - 1990
    print("starting download w/ 1980-1990....")
    curr_params = {'start_year':str(1980), 'end_year':str(1990)}
    ads_url = build_url(curr_params)

    obtain_ads_data(ads_url, 0)

    print("Done with 1980-1990")

    # 1 == 1991 - 2000
    curr_params = {'start_year':str(1991), 'end_year':str(2000)}
    ads_url = build_url(curr_params)
    obtain_ads_data(ads_url, 1)

    print("Done with 1991-2000")

    # 2 == 2001 - 2010
    curr_params = {'start_year':str(2001), 'end_year':str(2010)}
    ads_url = build_url(curr_params)
    obtain_ads_data(ads_url, 2)

    print("Done with 2001-2010")

    # 3 == 2011 - 2021
    curr_params = {'start_year':str(2011), 'end_year':str(2021)}
    ads_url = build_url(curr_params)
    obtain_ads_data(ads_url, 3)

    print("Done with 2011-2021")

if __name__ == '__main__':
    main()
