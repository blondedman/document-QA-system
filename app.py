import os
import requests
import tempfile
import streamlit as st
 
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

@st.cache_resource
def get_embeddings_model():
    """loads the HuggingFace embeddings model and caches it."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_qa_pipeline():
    """loads the QA pipeline and caches it."""
    model_name = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return pipeline("question-answering", model=model, tokenizer=tokenizer)


def process_document(document_path):
    """
    loads a PDF document and splits it into manageable text chunks
    
    args:
        document_path (str): file path to the PDF document.
        
    returns:
        list: a list of text chunks (documents).
    """
    # 1. load the document
    loader = PyPDFLoader(document_path)
    documents = loader.load()

    # 2. chunk the document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    texts = text_splitter.split_documents(documents)
    
    return texts


def create_vector_store(texts, embeddings):
    """
    creates a FAISS vector store from text chunks.
    
    args:
        texts (list): a list of text chunks.
        embeddings: the embeddings model.
        
    returns:
        FAISS: a FAISS vector store object.
    """
    db = FAISS.from_documents(texts, embeddings)
    return db

def convert_gdrive_link(url):
    """
    converts a Google Drive shareable link to a direct download link
    if the link is not a Google Drive link, it returns the original URL
    """
    try:
        # check if it's a Google Drive link
        if "drive.google.com" in url:
            # extract the file ID from the URL.
            # the ID is the part between 'd/' and the next '/'
            file_id = url.split('/d/')[1].split('/')[0]
            
            # construct the new direct download link
            new_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            return new_url
        else:
            # if not a G-Drive link, return original URL
            return url
            
    except IndexError:
        # if the URL format is unexpected, return the original URL
        return url
    
hide_input_instructions_css = """
    <style>
    div[data-testid="InputInstructions"] > span:nth-child(1) {
        visibility: hidden;
    }
    </style>
"""

st.markdown(hide_input_instructions_css, unsafe_allow_html=True)

st.title("📄 QA SYSTEM")
st.write("enter a URL to a PDF document and ask questions about its content")

if 'retriever' not in st.session_state:
    
    url = st.text_input("enter the google drive URL of the PDF document", key="pdf_url")
    
    if st.button("Process Document"):
        if url:
            with st.spinner("fetching and processing document... this may take a moment"):
                try:
                    # 1. fetch the document from the URL
                    url = convert_gdrive_link(url)
                    response = requests.get(url)
                    response.raise_for_status()  # Raise an exception for bad status codes

                    # 2. save the content to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(response.content)
                        tmp_file_path = tmp_file.name
                    
                    # 3. process the document using the existing function
                    embeddings = get_embeddings_model()
                    texts = process_document(tmp_file_path)
                    vector_store = create_vector_store(texts, embeddings)
                    st.session_state.retriever = vector_store.as_retriever(search_kwargs={'k': 5})
                    
                    # clean up the temporary file
                    os.remove(tmp_file_path)
                    
                    # rerun to update the app state and show the question input
                    st.rerun()

                except requests.exceptions.RequestException as e:
                    st.error(f"error fetching URL: {e}")
                except Exception as e:
                    st.error(f"an error occurred during processing: {e}")
        else:
            st.warning("please enter a valid URL")

else:
    # this block runs AFTER the retriever has been created and stored in session_state
    st.success("document processed successfully! you can now ask questions")

    # question input from the user
    question = st.text_input("ask a question about the document")

    if question:
        with st.spinner("finding the answer..."):
            retriever = st.session_state.retriever
            
            # 1. retrieve relevant documents
            docs = retriever.get_relevant_documents(question)
            
            # 2. combine documents into a single context string
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # 3. call the pipeline with the question and context
            qa_pipeline = get_qa_pipeline()
            result = qa_pipeline(question=question, context=context)
            
            # display the answer
            answer = result["answer"]
            score = result["score"]
        
            CONFIDENCE_THRESHOLD = 0.0005 
        
            st.subheader("answer:")
        
            if answer and score > CONFIDENCE_THRESHOLD:
                st.write(answer.lower())
                st.info(f"**confidence score:** {score:.2%}")
            else:
                st.warning("no answer found in the document for this question")


if st.button("Clear & Start Again"):
            st.session_state.clear()
            st.rerun()