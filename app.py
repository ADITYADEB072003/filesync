import os
import re
import pytesseract
from pdf2image import convert_from_path
from docx import Document
import openpyxl
from PIL import Image

def extract_text_pdf(path):
    try:
        pages = convert_from_path(path, 200)
        text = ""
        for page in pages:
            text += pytesseract.image_to_string(page)
        return text
    except Exception as e:
        return f"[PDF ERROR] {str(e)}"

def extract_text_docx(path):
    try:
        doc = Document(path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"[DOCX ERROR] {str(e)}"

def extract_text_xlsx(path):
    try:
        wb = openpyxl.load_workbook(path)
        text = ""
        for ws in wb.worksheets:
            for row in ws.iter_rows():
                row_text = " ".join([str(cell.value) if cell.value else "" for cell in row])
                text += row_text + "\n"
        return text
    except Exception as e:
        return f"[XLSX ERROR] {str(e)}"

def extract_text_image(path):
    try:
        return pytesseract.image_to_string(Image.open(path))
    except Exception as e:
        return f"[IMAGE ERROR] {str(e)}"

def index_directory(root_dir):
    index = []
    supported = (".pdf", ".docx", ".xlsx", ".xls", ".png", ".jpg", ".jpeg")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            ext = file.lower().split('.')[-1]
            path = os.path.join(root, file)
            try:
                if file.lower().endswith(".pdf"):
                    text = extract_text_pdf(path)
                elif file.lower().endswith(".docx"):
                    text = extract_text_docx(path)
                elif file.lower().endswith((".xlsx", ".xls")):
                    text = extract_text_xlsx(path)
                elif file.lower().endswith((".png", ".jpg", ".jpeg")):
                    text = extract_text_image(path)
                else:
                    continue
                index.append({'file': path, 'text': text})
            except Exception as e:
                index.append({'file': path, 'text': f"[ERROR] {e}"})
    return index

def clean_snippet(snippet):
    # Remove file paths from snippet for cleaner output
    snippet = re.sub(r'/?[\w\.-]+/?', ' ', snippet)
    snippet = re.sub(r'\s+', ' ', snippet).strip()
    return snippet

def search_index(index, keyword):
    results = []
    keyword_lower = keyword.lower()
    for entry in index:
        text_lower = entry["text"].lower()
        if keyword_lower in text_lower:
            pos = text_lower.find(keyword_lower)
            snippet = entry["text"][max(0,pos-40):pos+len(keyword)+40].replace('\n',' ')
            snippet = clean_snippet(snippet)
            results.append({"file": entry["file"], "snippet": snippet})
    return results

if __name__ == "__main__":
    folder = input("Enter folder to scan: ").strip()
    term = input("Enter search keyword: ").strip()
    print("Indexing files. Please wait...")
    index = index_directory(folder)
    print(f"Indexed {len(index)} files.")
    results = search_index(index, term)
    print(f"\nSearch results for '{term}':")
    if results:
        for result in results:
            print(f"\nFile: {result['file']}\n...{result['snippet']}...")
    else:
        print("No matches found.")
