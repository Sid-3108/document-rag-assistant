import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import os
from dotenv import load_dotenv
from anthropic import Anthropic


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Document Q&A",
    page_icon="📚",
    layout="centered"
)


# -----------------------------------
# TITLE
# -----------------------------------

st.title("Document Q&A using RAG - By Sid")

st.write(
    "Upload a PDF and ask questions about its content."
)


# -----------------------------------
# LOAD EMBEDDING MODEL
# -----------------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_embedding_model()


# -----------------------------------
# EXTRACT TEXT
# -----------------------------------

def extract_text_from_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------------
# SPLIT INTO CHUNKS
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
# UPLOAD PDF
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# -----------------------------------
# PROCESS PDF
# -----------------------------------

if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    with st.spinner("Processing document..."):

        # Extract text
        pdf_text = extract_text_from_pdf(
            uploaded_file
        )

        # Create chunks
        chunks = split_into_chunks(
            pdf_text
        )

        # Create embeddings
        embeddings = model.encode(
            chunks
        )

        # Create FAISS index
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(
            dimension
        )

        index.add(embeddings)

    st.success(
        f"Document processed successfully! "
        f"{len(chunks)} chunks created."
    )


    # -----------------------------------
    # QUESTION
    # -----------------------------------

    question = st.text_input(
        "Ask a question about the document:"
    )


    # -----------------------------------
    # ASK BUTTON
    # -----------------------------------

    if st.button("🔎 Ask"):

        if not question:

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching document and generating answer..."
            ):

                # Create question embedding
                question_embedding = model.encode(
                    [question]
                )

                # Retrieve top 3 chunks
                distances, indices = index.search(
                    question_embedding,
                    k=3
                )

                # Build context
                retrieved_chunks = []

                for index_number in indices[0]:

                    retrieved_chunks.append(
                        chunks[index_number]
                    )

                context = "\n\n".join(
                    retrieved_chunks
                )


                # -----------------------------------
                # CLAUDE
                # -----------------------------------

                load_dotenv()

                api_key = os.getenv(
                    "ANTHROPIC_API_KEY"
                )

                client = Anthropic(
                    api_key=api_key
                )


                prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY
the information provided in the document context.

If the answer cannot be found in the context,
say:

"I couldn't find the answer in the document."

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""


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


                answer = message.content[0].text


            # -----------------------------------
            # DISPLAY ANSWER
            # -----------------------------------

            st.subheader("💬 Answer")

            st.write(answer)


            # -----------------------------------
            # SHOW RETRIEVED SOURCES
            # -----------------------------------

            with st.expander(
                "🔎 View retrieved document sections"
            ):

                for i, index_number in enumerate(
                    indices[0]
                ):

                    st.markdown(
                        f"**Retrieved section {i + 1}**"
                    )

                    st.write(
                        chunks[index_number]
                    )

                    st.divider()