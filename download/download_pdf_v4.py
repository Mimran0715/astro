import sys
from urllib.parse import urlencode, quote_plus
import requests
#from os import listdir, mkdir
import os
from urllib.request import urlretrieve
import sqlite3
#import text_process as t
#import sqlite_db
import pdfplumber as p
import time
import sys
import arxiv
import json
import ads

API_TOKEN = "4EqqwYzTgJYNypXmhwVZNA4UiDytgRdqbcxURdD9"

#Building the url to access the NASA ADS API with a year range , ex: year: 1999-2000
def build_url(params:dict) -> str:
    # params is currently : {start_year:value, end_year:value}
    base_url = "https://api.adsabs.harvard.edu/v1/search/query?q="
    query = "year:" + params['start_year'] + "-" + params['end_year']
    encoded_query = quote_plus(query)
    url = base_url + encoded_query + "&fq=database:astronomy&" + "fl=id,links_data,bibcode,doi,arxiv_class,title,year,author,abstract,citation_count,identifier&" #+ "rows=2000"
    #url = base_url + encoded_query + "&fq=database:astronomy&" + "rows=2000"
    return url

def query_ads(query):
    pass

# getting the text of the pdf
def obtain_pdf_text(path:str):
    with p.open(path) as pdf:
        #print("hello")
        #print(pdf.pages)
        paper_text = ""
        for i in range(len(pdf.pages)):
            text = pdf.pages[i].extract_text(x_tolerance=1, y_tolerance=3)
            paper_text += text
        return paper_text
        #print("Time taken to extract text", time.time() - t0)
        #print(paper_text)
        

# Obtaining the ADS response and extracting the pdf url (related to Arxiv) and retrieving the pdf
def obtain_ads_data(url:str, db: int, path:str, db_path:str, table_name:str) -> None:
    ads_response= requests.get(url,headers={'Authorization': 'Bearer ' + API_TOKEN}).json()
    #print(type(ads_response['response']['numFound']))
    #print(ads_response.keys())
    #return
    # with open('data.txt', 'w') as of:
    #     json.dump(ads_response, of)

    #print(ads_response['response'].keys())
    print(ads_response['response']['start'])
    return
    papers = ads_response['response']['docs']
    print(papers)
    return
    print(len(papers))
    #for paper in papers:

    # folder_count = 1
    # missing_count = 0
    # total_count = 0
    # request_count = 0
    # paper_count = 0
    # no_pdf_count = 0
    # print("RESPONSE RESPONSE RESPONSE")
    # print(ads_response['response'])
    # print()
    # print()
    # print()
    #print(ads_response['response']['numFound'])
    #return

    
    # while ads_response['response']['numFound'] != 0:

    #     papers = ads_response['response']['docs']
    #     #print("PAPERS PAPERS PAPERS")
    #     #print(len(papers))
    #     #count  = 0
    #     # for p in papers:
    #     #     if count == 1:
    #     #         break
    #     #     #print(p)
    #     #     count += 1 
            
    #     #break
    #     for paper in papers:
    #         print("ID:", paper['id'])
    #         print("Title:", paper['title'])
    #         #break
    #         folder_path = path + str(folder_count)
    #         if(os.path.exists(folder_path)==0):
    #             os.mkdir(folder_path)
    #         if paper_count == 5000:
    #             # make a new folder
    #             folder_count +=1
    #             folder_path = path + str(folder_count)
    #             os.mkdir(folder_path)   
    #             paper_count = 0

    #         # file_name = folder_path + '/' + paper['title'][0] + ".pdf"
    #         # try:
    #         #     x = 0
    #         #     print("Links Data")
    #         #     urls = []
    #         #     for link in paper['links_data']:
    #         #         link_dict = json.loads(link)
    #         #         urls.append(link_dict['url'])

    #         #     #print(urls)

    #         #     try:
    #         #         pub_pdf =  "https://ui.adsabs.harvard.edu/link_gateway/" + paper['bibcode'] + "/PUB_PDF" 
    #         #         #eprint_pdf = "https://ui.adsabs.harvard.edu/link_gateway/" + paper['bibcode'] + "/EPRINT_PDF" 
    #         #         # for url in urls:
    #         #         #     r = requests.get(url)
    #         #         #     print(r.url)
    #         #         #     print(r.history)
    #         #         #     #urlretrieve(url, file_name)
    #         #         #     print("am here")
    #         #         r = requests.get(pub_pdf)
    #         #         print(r.url)
    #         #         print(r.history)
    #         #         urlretrieve(url, file_name)
    #         #     except:
    #         #         print("pub pdf issue")
    #         #         x = 1
                
    #         #     try:
    #         #         #pub_pdf =  "https://ui.adsabs.harvard.edu/link_gateway/" + paper['bibcode'] + "/PUB_PDF" 
    #         #         eprint_pdf = "https://ui.adsabs.harvard.edu/link_gateway/" + paper['bibcode'] + "/EPRINT_PDF" 
    #         #         # for url in urls:
    #         #         #     r = requests.get(url)
    #         #         #     print(r.url)
    #         #         #     print(r.history)
    #         #         #     #urlretrieve(url, file_name)
    #         #         #     print("am here")
    #         #         r = requests.get(eprint_pdf)
    #         #         print(r.url)
    #         #         print(r.history)
    #         #         urlretrieve(url, file_name)
    #         #     except:
    #         #         print("eprint pdf issue")
    #         #         x = 2
                
    #         #     if(x == 2):
    #         #         print('Both issues')

    #         # except KeyError:
    #         #     print("Missing Links Data")
    #         # try:
    #         #     print("Arxiv Class")
    #         #     print(paper['arxiv_class'])
    #         # except KeyError:
    #         #     print("Missing Arxiv Class")
            
    #         # print("Identifiers")
    #         # print(paper['identifier'])

            
    #         # print()
    #         # search = arxiv.Search(query="astro", max_results = 10)
    #         # print(search)
    #         # for result in search.get():
    #         #     print(result.title)
    #         # print()
    #         # print("search 2", paper['doi'])
    #         # search_2 = arxiv.Search(query=paper['doi'], max_results = 10)
    #         # print(search_2)
    #         # for result in search_2.get():
    #         #     print(result.title)
    #         # print()
    #         #return
    #         #import json
    #         # print("link data", paper['links_data'])
    #         # d = json.loads(paper['links_data'][0])
    #         # u = d['url']
    #         # print("U", u)
    #         # pdf_url = "https://ui.adsabs.harvard.edu/link_gateway/" + paper['bibcode'] + "/EPRINT_PDF" 
    #         # pdf_url_2 = "https://ui.adsabs.harvard.edu/link_gateway/" + paper['bibcode'] + "/PUB_PDF" 
    #         # #print(type(paper['bibcode']))
    #         # #print("https://ui.adsabs.harvard.edu/link_gateway/" + str(paper['bibcode']) + '/EPRINT_PDF')
    #         # #folder_path = '/Users/Mal/Desktop/research/' # change file string based on how many files in folder, < 50,000
    #         # #folder_path = '/home/maleeha/research/' # change file string based on how many files in folder, < 50,000
    #         # pdf_url_3 = "https://arxiv.org/pdf/2009.00263.pdf"
            
    #         # # try:
    #         # print("in try")
    #         # pd_response = requests.get(u, headers={'Authorization': 'Bearer ' + API_TOKEN})
    #         # print(pd_response.url)
    #         # print(pd_response.history)
    #         # file_name = folder_path + '/' + paper['title'][0] + ".pdf"
    #         # if 'abstract' in pd_response.url:
    #         #     a = pd_response.url.replace("abstract", "pdf")
    #         # resp = requests.get(a, headers={'Authorization': 'Bearer ' + API_TOKEN})
    #         # urlretrieve(pdf_url, file_name)
    #         # #print(pd_response.json())
    #         # print("DONEONISFJKD:JFLKSD")
    #         #     # break
    #         # print('otehr one')
    #         # pdf_response= requests.get(pdf_url,headers={'Authorization': 'Bearer ' + API_TOKEN})
    #         # #return (void *)((pt[ptindex] & ~0xFFF) + ((unsigned long)virtualaddr & 0xFFF));
    #         # print(pdf_response.url)
    #         # print(pdf_response.history)

    #         # print('other one oterhonedsjdjs;fkdsfs')
    #         # pdf_response_2= requests.get(pdf_url_2,headers={'Authorization': 'Bearer ' + API_TOKEN})
    #         # #print(pdf_response_2.json())
    #         # print(pdf_response_2.url)
    #         # print(pdf_response_2.history)
    #         # return
            
    #         # #print(pdf_url, file_name)
            
    #         # # except:
    #         # #     print("not working")
    #         # #     print()
    #         # #     print("PDF URL 1 ", pdf_url)
    #         # #     #print("json 1", pdf_response.json())
    #         # #     #print("filename", file_name)
    #         # #     print("PDF URL 2", pdf_url_2)
    #         # #     print()
    #         # #     no_pdf_count += 1
    #         # #     print("Current no pdf count", no_pdf_count)
    #         # #x = requests.get(pdf_url)
    #         # #print(x)
    #         # # input data into the database 
    #         # # file_string = file_name
    #         # # #text = t.obtain_pdf_text(file_string)
    #         # # #break
    #         # # try:
    #         # #     paper_text = obtain_pdf_text(file_name)
    #         # # except:
    #         # #     print('cant download pdf')
    #         # #     print()
    #         # break
    #         # #print(paper['author'])
    #         try:
    #             if db == 0:
    #                 db_command(db_path, "INSERT INTO " + table_name + " (bibcode,title,year,author,abstract,citation_count,file_path,downloaded_pdf, ran_sentiment,\
    #                     sentiment, paper_text, word_vector) VALUES (?,?,?,?,?,?,?,?,?,?,?,?);", (paper['bibcode'], paper['title'][0], int(paper['year']), \
    #                         ' '.join(paper['author']), paper['abstract'], int(paper['citation_count']), file_string, 1, 0, 0.0, text, None))
            
    #                 #print('Inserted pdf') # dont know if it should be None above for blob
    #             # elif db == 1:
    #             #     db_command(db_path, '''INSERT INTO astro_papers_1991_2000 (bibcode,title,year,author,abstract,citation_count,file_path,downloaded_pdf, ran_sentiment,\
    #             #         sentiment, paper_text, word_vector) VALUES (?,?,?,?,?,?,?,?,?,?,?,?);''', (paper['bibcode'], paper['title'][0], int(paper['year']), \
    #             #             ' '.join(paper['author']), paper['abstract'], int(paper['citation_count']), file_string, 1, 0, 0.0, text, None))
            
    #             #     #print('Inserted pdf') # dont know if it should be None above for blob
    #             # elif db == 2:
    #             #     db_command(db_path, '''INSERT INTO astro_papers_2001_2010 (bibcode,title,year,author,abstract,citation_count,file_path,downloaded_pdf, ran_sentiment,\
    #             #         sentiment, paper_text, word_vector) VALUES (?,?,?,?,?,?,?,?,?,?,?,?);''', (paper['bibcode'], paper['title'][0], int(paper['year']), \
    #             #             ' '.join(paper['author']), paper['abstract'], int(paper['citation_count']), file_string, 1, 0, 0.0, text, None))
            
    #             #     #print('Inserted pdf') # dont know if it should be None above for blob
    #             # elif db == 3:
    #             #     db_command(db_path, '''INSERT INTO astro_papers_2011_2021 (bibcode,title,year,author,abstract,citation_count,file_path,downloaded_pdf, ran_sentiment,\
    #             #         sentiment, paper_text, word_vector) VALUES (?,?,?,?,?,?,?,?,?,?,?,?);''', (paper['bibcode'], paper['title'][0], int(paper['year']), \
    #             #             ' '.join(paper['author']), paper['abstract'], int(paper['citation_count']), file_string, 1, 0, 0.0, text, None))
    #         except:
    #             print('aint working')
    #         # except KeyError:
    #         #     missing_count +=1
    #         #     print("missing a value - skipping this one, not entering into sqlite db")
    #         total_count +=1
    #         paper_count +=1
    #     request_count +=1 
    #     print("request {r} done...".format(r=request_count))
    #     if(request_count == 3):
    #         print("3 requests done, now moving on to next year bracket")
    #         break
    #     time.sleep(2)
    #     #ads_response= requests.get(url,headers={'Authorization': 'Bearer ' + API_TOKEN}).json()
    # print("Missing number of papers bc of KeyError: ", missing_count)
    # print("Papers gotten: ", total_count - missing_count)

# Helper function to update database
def db_command(path, command, task):
     #conn = sqlite3.connect('/home/maleeha/research/research.db')
     conn = sqlite3.connect(path)
     #conn = sqlite3.connect('/home/maleeha/research/data/research.db')
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
    # db_path = sys.argv[1]
    # folder_path = sys.argv[2]
    # table_name = sys.argv[3]

    db_path = '/Users/Mal/Documents/research.db'
    folder_path = '/Users/Mal/Documents/pdfs/'

    run_loc = sys.argv[1] # local == 0, clotho == 1
    table_name = sys.argv[2]

    if run_loc == 1:
        db_path = '/home/maleeha/research/research.db'
        folder_path = '/home/maleeha/research/pdfs/'

    elif run_loc == 2:
        db_path = '/Users/Mal/Desktop/t.db'
        folder_path = '/Users/Mal/Desktop/pdfs/'

    if(os.path.exists(folder_path)==0):
        os.mkdir(folder_path)

    print("starting download w/ 1980-1990....")
    curr_params = {'start_year':str(1980), 'end_year':str(1990)}
    ads_url = build_url(curr_params)

    obtain_ads_data(ads_url, 0, folder_path, db_path, table_name)

    print("Done with 1980-1990")
    return
    # 1 == 1991 - 2000
    curr_params = {'start_year':str(1991), 'end_year':str(2000)}
    ads_url = build_url(curr_params)
    obtain_ads_data(ads_url, 1, folder_path, db_path, table_name)

    print("Done with 1991-2000")

    # 2 == 2001 - 2010
    curr_params = {'start_year':str(2001), 'end_year':str(2010)}
    ads_url = build_url(curr_params)
    obtain_ads_data(ads_url, 2, folder_path, db_path, table_name)

    print("Done with 2001-2010")

    # 3 == 2011 - 2021
    curr_params = {'start_year':str(2011), 'end_year':str(2021)}
    ads_url = build_url(curr_params)
    obtain_ads_data(ads_url, 3, folder_path, db_path, table_name)

    print("Done with 2011-2021")

if __name__ == '__main__':
    main()
