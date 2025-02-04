# Step 1: Import libraries
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import RetrievalQA

#Step 2: Streamlit File Uploader
##Streamlit file uploader for CSV files
# Streamlit file uploader for CSV files
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Save PDF temporarily
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
        
    #Step 3: Load PDF file and create Embeddings
    loader = PyPDFLoader(file_path=temp_file_path)
    docs = loader.load()
    embeddings = HuggingFaceEmbeddings()

    #Step 4: Create vector Store
    vector_store = FAISS.from_documents(docs, embeddings)

    #Step 5: Connect a retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    #Step 6: Define the LLM
    llm = Ollama(model="deepseek-r1:1.5b")  # Our 1.5B parameter model

    #Step 7: Prompt
    prompt = """
        1. Use ONLY the context below.
        2. If unsure, say "I don't know".
        3. Keep answers under 4 sentences.

        Context: {context}

        Question: {question}

        Answer:
        """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)
        
    # Step 8: Define the QA Chain
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)
        
    # Combine document chunks
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=QA_CHAIN_PROMPT
    )
        
    #Step 9: Create the RetrievalQA Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff", 
    )

    # Step 10: Create Streamlit UI for the application
    user_input = st.text_input("Ask your PDF a question:")

    if user_input:
        with st.spinner("Thinking..."):
            try:
                response = qa.run(user_input)  
                st.write(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")
            