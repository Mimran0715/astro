
import pdfplumber as p
import time
import sys

def main():
    path = sys.argv[1]
    with p.open(path) as pdf:
        #print("hello")
        #print(pdf.pages)
        paper_text = ""
        t0 = time.time()
        for i in range(len(pdf.pages)):
            text = pdf.pages[i].extract_text(x_tolerance=1, y_tolerance=3)
            paper_text += text
    print(paper_text)
    print("Time taken to extract text", time.time() - t0)

if __name__ == "__main__":
    main()