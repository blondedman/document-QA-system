# 📄 Document Question Answering System

a simple yet powerful streamlit application that lets you link a **Google Drive-Hosted PDF**, processes its contents using **HuggingFace embeddings**, and allows you to ask natural language questions based on the document. the app uses **LangChain**, **FAISS**, and **transformers QA pipeline** to find the best possible answers from the document.

## FEATURES
- 🧠 semantic understanding of PDF content using **sentence-transformers**
- 🔍 document chunking and vector search with **FAISS**
- 🤖 question answering with **deepset/roberta-base-squad2**
- 🔗 accepts **Google Drive URLs** or others
- 💾 efficient processing using **@st.cache_resource**
- 🧹 clean, user-friendly interface built with **streamlit**

## 📦 DEPENDENCIES
install the required packages using:
```bash
pip install streamlit langchain transformers sentence-transformers faiss-cpu requests
