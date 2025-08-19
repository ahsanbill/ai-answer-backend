# utils/chunking.py
import fitz
import re

def hierarchical_chunking(pdf_path):
    doc = fitz.open(pdf_path)
    section_chunks = []
    paragraph_chunks = []

    section_buffer = []

    def save_section():
        if section_buffer:
            content = "\n\n".join(section_buffer)
            section_chunks.append(content)
            section_buffer.clear()

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            # Skip non-text blocks
            if "lines" not in b:
                continue

            for l in b["lines"]:
                for s in l["spans"]:
                    text = s["text"].strip()
                    if not text:
                        continue

                    # Detect headings (size-based or numbered headings)
                    if s.get("size", 0) > 14 or re.match(r"^\d+(\.\d+)*\s+[A-Z]", text):
                        save_section()
                        section_buffer.append(text)
                    else:
                        section_buffer.append(text)
                        paragraph_chunks.append(text)

    save_section()
    return section_chunks, paragraph_chunks
