import pdfplumber
import PyPDF2
import io

class PDFParser:
    def __init__(self):
        pass

    def extract_text_pdfplumber(self, file_path_or_bytes):
        """Extracts text using pdfplumber, better for tables and layout."""
        text = ""
        try:
            with pdfplumber.open(file_path_or_bytes) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error reading with pdfplumber: {e}")
        return text.strip()

    def extract_text_pypdf2(self, file_path_or_bytes):
        """Fallback extraction using PyPDF2."""
        text = ""
        try:
            reader = PyPDF2.PdfReader(file_path_or_bytes)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
             print(f"Error reading with PyPDF2: {e}")
        return text.strip()

    def parse_resume(self, file_path_or_bytes):
        """Attempts to extract text using multiple methods until successful."""
        text = self.extract_text_pdfplumber(file_path_or_bytes)
        if not text:
            # Fallback
            text = self.extract_text_pypdf2(file_path_or_bytes)
            
        return text
