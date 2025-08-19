import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from config.settings import settings

class VectorStoreService:
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
        self.vector_folder = settings.VECTOR_FOLDER

    def build_store(self, paragraphs_with_meta, headings, subheadings):
        texts, metadatas = [], []

        # Store paragraphs
        for idx, p in enumerate(paragraphs_with_meta):
            texts.append(p["text"])
            metadatas.append({
                "type": "paragraph",
                "heading": p.get("heading", ""),
                "subheading": p.get("subheading", ""),
                "index": idx
            })

        # Store headings
        for h in headings:
            texts.append(h)
            metadatas.append({
                "type": "heading",
                "heading": h,
                "subheading": "",
                "index": None
            })

        # Store subheadings
        for sh in subheadings:
            texts.append(sh)
            metadatas.append({
                "type": "subheading",
                "heading": "",
                "subheading": sh,
                "index": None
            })

        store = FAISS.from_texts(texts, self.embedding_model, metadatas=metadatas)
        store.save_local(self.vector_folder)
        return store

    def load_store(self):
        return FAISS.load_local(
            self.vector_folder,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

    def exists(self):
        return os.path.exists(os.path.join(self.vector_folder, "index.faiss"))
