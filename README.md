# PDF Chatbot with Streamlit

This repository contains a PDF Chatbot application built with Streamlit, LangChain, Google Generative AI, FAISS, and other libraries. The application allows users to upload PDF documents, and then ask questions about the content in the PDFs. The chatbot will use Google Generative AI and FAISS vector store to provide accurate answers based on the provided PDF content.

**Please note:** This application is specifically designed to work with [this PDF](https://drive.google.com/file/d/1NP51IbpiXiVl0omdD5KrdKmNDn_ZSdpN/view?usp=sharing). 

## Streamlit App

You can access the deployed Streamlit app [here](https://pdf-chatbot-llm.streamlit.app/).

## Installation

To run the application locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/pdf-chatbot.git
    cd pdf-chatbot
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory and add your Google API key:
    ```plaintext
    GOOGLE_API_KEY=your_google_api_key
    ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. In the application, you can upload PDF files through the sidebar and submit them to process the text.

3. After processing, you can interact with the chatbot by asking questions related to the PDF content.

4. Use the 'Clear Chat' button in the sidebar to reset the chat history.

## Code Overview

The application consists of several key components:

- **`app.py`**: The main script that runs the Streamlit application.
- **`get_pdf_text()`**: Function to extract text from uploaded PDF files.
- **`get_text_chunks()`**: Function to split the text into smaller chunks.
- **`get_vector_store()`**: Function to create a FAISS vector store from text chunks.
- **`get_conversational_chain()`**: Function to create a conversational chain for question answering.
- **`user_input()`**: Function to handle user questions and generate responses using the conversational chain.

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain): The framework for building the application.
- [Google Generative AI](https://ai.google/): Used for embeddings and the conversational chain.
- [Streamlit](https://streamlit.io/): The framework for building the web application.

## Contact

For any questions or feedback, feel free to open an issue on GitHub or reach out to the repository owner.
