from pypdf import PdfReader
import sys

pdf_path = r"docs\s41598-025-93376-9.pdf"
reader = PdfReader(pdf_path)
query_terms = ["SANGO", "Northern Goshawk", "Self-Adaptive", "prey", "dynamic factor", "DF", "prey identification", "prey capture"]

found = []
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ''
    for term in query_terms:
        if term.lower() in text.lower():
            # print surrounding context
            lines = text.splitlines()
            for li, line in enumerate(lines):
                if term.lower() in line.lower():
                    start = max(0, li-4)
                    end = min(len(lines), li+5)
                    context = "\n".join(lines[start:end])
                    found.append((i+1, term, context))

if not found:
    print("No hits for query terms.")
    sys.exit(0)

for page, term, ctx in found:
    print(f"--- Page {page} (term: {term}) ---")
    print(ctx)
    print("\n")

