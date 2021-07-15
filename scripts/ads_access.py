import requests
import PyPDF2

from bs4 import BeautifulSoup

start_year = 0
end_year = 0

token = "4EqqwYzTgJYNypXmhwVZNA4UiDytgRdqbcxURdD9"

#r = requests.get("link", params = {}, headers={}, data = )

r = requests.get("https://api.adsabs.harvard.edu/v1/search/query?q=author%3Amart%C3%ADnez+neutron+star&fl=author&rows=1",\
               headers={'Authorization': 'Bearer ' + token})

#print(r.json()) 

'''
from urllib.request import urlopen, urlretrieve
url = 'http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=1'
data = urlopen(url).read()

encoding = "utf-8"
print(type(data))

x = data.decode(encoding)

#print(type(x))

#print(x)

soup = BeautifulSoup(x, 'xml')
titles = soup.find_all('title')
links = soup.find_all('link')
#for title in titles:
#    print(title.get_text())
for link in links:
    if 'title' in link.attrs and link['title'] == 'pdf':
        print(link)


y = urlretrieve(link['href'], "filename.pdf") 
pdf_file_object = open("filename.pdf", 'rb')

pdf_reader = PyPDF2.PdfFileReader(pdf_file_object)

for page in range(pdf_reader.numPages):
    page_object = pdf_reader.getPage(page)
    print(page_object.extractText())

'''