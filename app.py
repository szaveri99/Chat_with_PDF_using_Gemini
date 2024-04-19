# import the required libraries

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Load environment variables from a .env file
load_dotenv()
# Get the Google API key from the environment variables
api_key = os.getenv("GOOGLE_API_KEY")
# Configure the Google Generative AI with the API key
genai.configure(api_key=api_key)

# Function to read text from PDF files
def get_pdf_text(pdf_docs):
    text = ""  # Initialize an empty string to hold the text
    for pdf in pdf_docs:
        # Create a PDF reader object for each PDF file
        pdf_reader = PdfReader(pdf)
        # Extract text from each page of the PDF
        for page in pdf_reader.pages:
            text += page.extract_text()  # Append the text from the page to the string
    return text

# Function to split the text into chunks
def get_text_chunks(text):
    # Initialize a text splitter with a specified chunk size and overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000
    )
    # Split the text into chunks and return as a list of strings
    chunks = splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(chunks):
    # Create embeddings using the Google Generative AI model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Create a FAISS vector store from the text chunks and embeddings
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    # Save the vector store locally for future use
    # vector_store.save_local("./faiss_index")

# Function to create a conversational chain for question answering
def get_conversational_chain():
    # Define a prompt template for the question-answering chain
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, say, 'answer is not available in the context' and don't provide an incorrect answer.
    Context: {context}
    Question: {question}
    Answer:
    """
    # Initialize the ChatGoogleGenerativeAI model with specific parameters
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        client=genai,
        temperature=0.3,
    )
    # Create a prompt template for the question-answering chain
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    # Load the question-answering chain with the language model and prompt
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

# Function to clear chat history
def clear_chat_history():
    # Reset the chat history to its initial state
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some PDFs and ask me a question."}
    ]

# Function to handle user input and provide responses
def user_input(user_question):
    # Create embeddings using the Google Generative AI model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    # Load the FAISS vector store from the local file with dangerous deserialization enabled
    new_db = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
    # Perform a similarity search in the vector store using the user's question
    docs = new_db.similarity_search(user_question)
    # Get the conversational chain for question answering
    chain = get_conversational_chain()
    # Get the response by querying the chain with the relevant documents and user's question
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    return response

# Main function to run the Streamlit application
def main():
    # Set page configurations for Streamlit
    st.set_page_config(page_title="PDF Chatbot" )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu")
        # File uploader for multiple PDF files
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on Submit",
            accept_multiple_files=True
        )
        # Button to submit and process the uploaded PDF files
        if st.button("Submit"):
            # Display a spinner while processing
            with st.spinner("In Process..."):
                # Extract text from the uploaded PDF files
                raw_text = get_pdf_text(pdf_docs)
                # Split the text into chunks
                text_chunks = get_text_chunks(raw_text)
                # Create a vector store from the text chunks
                get_vector_store(text_chunks)
                st.success("Successful!!")

    # Main content area for displaying chat messages
    st.title("Chat with PDF 🤖")
    
    # Add a button to clear the chat history
    st.sidebar.button('Clear Chat', on_click=clear_chat_history)

    # Chat input and chat history
    if "messages" not in st.session_state.keys():
        # Initialize chat history with an introductory message
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question by uploading the pdf files"}
        ]

    # Display chat messages in the chat area
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle user input
    if prompt := st.chat_input():
        # Add the user's prompt to the chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":

        # Create a chat message for the assistant and display a spinner while thinking
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get the user's input
                response = user_input(prompt)
                # Placeholder for displaying the full response incrementally
                placeholder = st.empty()
                full_response = ''
                # Iterate through the response and display it incrementally
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                # Display the full response
                placeholder.markdown(full_response)

        # Add the assistant's response to the chat history
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)

# Entry point for the script
if __name__ == "__main__":
    main()
