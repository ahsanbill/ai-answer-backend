from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def split_text(text: str) -> list[str]:
    """
    Splits the text into full paragraphs.
    Paragraphs are separated by two or more newlines.
    Cleans up extra spaces.
    """
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split on two or more newlines
    paragraphs = re.split(r"\n\s*\n", text)

    # Clean and remove empty paragraphs
    clean_paragraphs = [
        para.strip()
        for para in paragraphs
        if para.strip()
    ]

    return clean_paragraphs