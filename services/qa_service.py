# services/qa_service.py
import re
from collections import defaultdict
from rank_bm25 import BM25Okapi
from openai import OpenAI
from config.settings import settings

class QAService:
    def __init__(self, vector_store, all_chunks):
        self.vector_store = vector_store
        self.all_chunks = all_chunks  # [{text, heading, subheading}]
        self.paras_norm = [self._normalize(p["text"]) for p in all_chunks]
        tokenized = [self._tokenize(t) for t in self.paras_norm]
        self.bm25 = BM25Okapi(tokenized)

        self.index_by_text = defaultdict(list)
        for i, p in enumerate(all_chunks):
            self.index_by_text[p["text"]].append(i)

        # for heading/subheading search
        self.headings = set(
            (p["heading"] or "").strip().lower() for p in all_chunks if p.get("heading")
        )
        self.subheadings = set(
            (p["subheading"] or "").strip().lower() for p in all_chunks if p.get("subheading")
        )

        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    # ---------- helpers ----------
    def _normalize(self, text: str) -> str:
        if not text:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip().lower()

    def _tokenize(self, text: str):
        return re.findall(r"\w+", text.lower())

    def _expand_query(self, query: str):
        return query

    def _minmax(self, pairs):
        if not pairs:
            return {}
        vals = [s for _, s in pairs]
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return {i: 0.0 for i, _ in pairs}
        return {i: (s - lo) / (hi - lo) for i, s in pairs}

    def _phrase_bonus(self, para_norm: str, q_norm: str, q_tokens: set[str]) -> float:
        bonus = 0.0
        if para_norm.startswith(q_norm):
            bonus += 2.0
        elif q_norm and q_norm in para_norm:
            bonus += 1.0
        if q_tokens:
            p_tokens = set(self._tokenize(para_norm))
            overlap = len(p_tokens & q_tokens) / max(1, len(q_tokens))
            bonus += 0.5 * overlap
        return bonus

    # ---------- heading/subheading match ----------
    def _match_section(self, question: str):
        """
        Try to match query against headings/subheadings.
        Returns all chunks if a section is matched confidently.
        """
        q_norm = self._normalize(question)

        # Check for explicit "section" mentions
        if "section" in q_norm or "chapter" in q_norm or "describe" in q_norm or "summarize" in q_norm:
            search_terms = self._tokenize(q_norm)
            # simple fuzzy match against headings/subheadings
            for h in self.headings:
                if all(t in h for t in search_terms if len(t) > 3):
                    return [c for c in self.all_chunks if self._normalize(c["heading"] or "") == h]
            for sh in self.subheadings:
                if all(t in sh for t in search_terms if len(t) > 3):
                    return [c for c in self.all_chunks if self._normalize(c["subheading"] or "") == sh]

        # Otherwise check semantic overlap with headings/subheadings
        tokens = set(self._tokenize(q_norm))
        best_match = None
        best_overlap = 0.0

        for h in list(self.headings) + list(self.subheadings):
            if not h:
                continue
            h_tokens = set(self._tokenize(h))
            overlap = len(tokens & h_tokens) / max(1, len(tokens))
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = h

        if best_match and best_overlap > 0.45:  # threshold for "enough match"
            return [
                c for c in self.all_chunks
                if self._normalize(c.get("heading", "")) == best_match
                or self._normalize(c.get("subheading", "")) == best_match
            ]

        return None

    # ---------- main ----------
    def retrieve_top_chunks(self, question: str, use_expansion: bool = False):
        query_for_retrieval = self._expand_query(question) if use_expansion else question
        q_norm = self._normalize(question)
        q_tokens = set(self._tokenize(question))
        q_for_retrieval_tokens = self._tokenize(self._normalize(query_for_retrieval))

        bm25_scores_arr = self.bm25.get_scores(q_for_retrieval_tokens)
        top_bm25 = sorted(range(len(bm25_scores_arr)),
                          key=lambda i: bm25_scores_arr[i],
                          reverse=True)[:50]

        faiss_docs = self.vector_store.similarity_search_with_score(query_for_retrieval, k=30)
        faiss_pairs = []
        for doc, raw_score in faiss_docs:
            sim = -raw_score if raw_score < 0 else 1.0 / (1.0 + float(raw_score))
            text = doc.page_content
            idx = self.index_by_text.get(text, [None])[0]
            if idx is not None:
                faiss_pairs.append((idx, sim))

        candidate_idxs = set(top_bm25) | {i for i, _ in faiss_pairs}
        bm25_pairs = [(i, bm25_scores_arr[i]) for i in candidate_idxs]
        bm25_norm = self._minmax(bm25_pairs)
        faiss_norm = self._minmax(faiss_pairs)

        results = []
        for i in candidate_idxs:
            para_norm = self.paras_norm[i]
            w_bm25 = 0.6
            w_faiss = 0.4
            score = w_bm25 * bm25_norm.get(i, 0.0) + w_faiss * faiss_norm.get(i, 0.0)
            score += self._phrase_bonus(para_norm, q_norm, q_tokens)
            results.append((i, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [self.all_chunks[i] for i, _ in results[:3]]

    def answer(self, question: str):
        # First try to match a section (heading/subheading)
        section_chunks = self._match_section(question)
        if section_chunks:
            top_chunks = section_chunks
        else:
            top_chunks = self.retrieve_top_chunks(question)

        context_text = "\n\n".join(
            f"[Heading: {c['heading'] or 'N/A'} | Subheading: {c['subheading'] or 'N/A'}]\n{c['text']}"
            for c in top_chunks
        )

        life_lessons_summary = (
            "Here are the 10 Life Lessons from the book:\n"
            "1. The Life-Saving Magic of Tidying Up – Sandhurst uses strict routines, inspections, and repetition of small tasks to build discipline, self-control, and pride.\n"
            "2. The Way You Spend Your Day – Purpose, rest, discipline, and planning are vital for building effectiveness and turning big challenges into manageable steps.\n"
            "3. Know Your Threshold of Failure – Cadets are pushed to the threshold of failure to learn resilience, humility, teamwork and self-belief, realising that failure is forgivable, temporary, and integral to success.\n"
            "4. You Do not Have to be a Minimalist – Packing your Bergen – with accessibility, balance, compactness, preparation and discipline – is an analogy for life: strip back unnecessary burdens, prioritise what you need, and share the load.\n"
            "5. Harness the Power of Your Team – Like wolves and soldiers, teamwork, loyalty, morale, honesty, trust and purpose bind individuals into something greater, empowering leaders and teams to endure, adapt and succeed.\n"
            "6. How Do You React When Things Go Wrong? – When you get lost – in the hills or in life – return to the last waypoint, check your moral compass, and reset your path with courage, values and feedback.\n"
            "7. Trust in Your Own Judgement – In moments of crisis, take a Condor Moment – stay calm, take a knee, trust your training, and let cooler heads prevail.\n"
            "8. Fail to Prepare and Prepare to Fail – Through testing endurance, drilling under pressure, rehearsing ‘what ifs’ and empowering daring initiative, Sandhurst teaches cadets that preparation, adaptability and practice are the seeds of victory.\n"
            "9. The Standard You Walk Past – Build yourself in high standards do not settle for less. Going to extra mile to succeed. Be active, do not be passive.\n"
            "10. The Power of Example – Lead by example, integrity, trust, empowering others.\n"
        )

        system_prompt = (
            "You are a helpful assistant answering questions strictly based on the provided book excerpts. "
            "The author is Major General Paul Nanson and the title is "
            "'Stand Up Straight - 10 Life Lessons from the Royal Military Academy Sandhurst'. "
            "Do NOT use external knowledge. If the answer is not in the context, say so clearly.\n\n"
            f"{life_lessons_summary}"
        )

        user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer strictly from the context above."

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )

        return {
            "answer": response.choices[0].message.content,
            "chunks_used": top_chunks
        }
