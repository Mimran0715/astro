import pdfplumber as p
import time


t0 = time.time()
with p.open("/Users/Mal/Documents/astro.pdf") as pdf:
    #print("hello")
    #print(pdf.pages)
    paper_text = ""
    for i in range(len(pdf.pages)):
        text = pdf.pages[i].extract_text(x_tolerance=1, y_tolerance=3)
        paper_text += text
    print("Time taken to extract text", time.time() - t0)
    print(paper_text)

    