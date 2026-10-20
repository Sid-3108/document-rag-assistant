from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import os
from dotenv import load_dotenv
from anthropic import Anthropic


# -----------------------------------
# 1. EXTRACT TEXT FROM PDF
# -----------------------------------

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------------
# 2. SPLIT TEXT INTO CHUNKS
# -----------------------------------

def split_into_chunks(text, chunk_size=500, overlap=50):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# -----------------------------------
# 3. LOAD PDF
# -----------------------------------

pdf_text = extract_text_from_pdf("documents/sample.pdf")

print(f"Total characters: {len(pdf_text)}")


# -----------------------------------
# 4. CREATE CHUNKS
# -----------------------------------

chunks = split_into_chunks(pdf_text)

print(f"Total chunks: {len(chunks)}")


# -----------------------------------
# 5. LOAD EMBEDDING MODEL
# -----------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")


# -----------------------------------
# 6. CREATE EMBEDDINGS
# -----------------------------------

embeddings = model.encode(chunks)

print(f"Embeddings created: {len(embeddings)}")


# -----------------------------------
# 7. CREATE FAISS INDEX
# -----------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print(f"FAISS vectors stored: {index.ntotal}")


# -----------------------------------
# 8. LOAD CLAUDE API KEY
# -----------------------------------

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")

client = Anthropic(api_key=api_key)


# -----------------------------------
# 9. ASK USER A QUESTION
# -----------------------------------

question = input("\nAsk a question about the PDF: ")


# -----------------------------------
# 10. EMBED THE QUESTION
# -----------------------------------

question_embedding = model.encode([question])


# -----------------------------------
# 11. RETRIEVE RELEVANT CHUNKS
# -----------------------------------

distances, indices = index.search(
    question_embedding,
    k=3
)


# -----------------------------------
# 12. BUILD CONTEXT
# -----------------------------------

retrieved_chunks = []

for index_number in indices[0]:
    retrieved_chunks.append(chunks[index_number])


context = "\n\n".join(retrieved_chunks)


# -----------------------------------
# 13. SEND CONTEXT + QUESTION TO CLAUDE
# -----------------------------------

prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information provided
in the document context below.

If the answer cannot be found in the context, say:
"I couldn't find the answer in the document."

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""


print("\nGenerating answer...")


message = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=500,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# -----------------------------------
# 14. DISPLAY FINAL ANSWER
# -----------------------------------

answer = message.content[0].text

print("\n====================================")
print("ANSWER")
print("====================================")

print(answer)