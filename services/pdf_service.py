import fitz
from collections import Counter

class PDFService:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract_with_structure(self):
        doc = fitz.open(self.pdf_path)
        results = {
            "headings": [],
            "subheadings": [],
            "paragraphs": []
        }

        font_sizes = []
        # Collect font sizes from all pages (for classification)
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    for l in b["lines"]:
                        for s in l["spans"]:
                            font_sizes.append(round(s["size"], 1))

        body_size = Counter(font_sizes).most_common(1)[0][0]
        heading_size = max(font_sizes)

        current_heading = None
        current_subheading = None

        # Iterate pages but skip first 4
        for page_index, page in enumerate(doc):
            if page_index < 4:
                continue

            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" not in b:
                    continue

                text = ""
                size = None

                for l in b["lines"]:
                    for s in l["spans"]:
                        text += s["text"] + " "
                        size = round(s["size"], 1)

                text = text.strip()
                if not text:
                    continue

                # Classification
                if size == heading_size:
                    current_heading = text
                    results["headings"].append(text)
                elif size > body_size:
                    current_subheading = text
                    results["subheadings"].append(text)
                else:
                    results["paragraphs"].append({
                        "text": text,
                        "heading": current_heading,
                        "subheading": current_subheading
                    })

        return results
