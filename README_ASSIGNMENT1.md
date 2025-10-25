To run my application with the provided set up: 
    - Essential prerequisites include: Python 3.11+ and the Cornell OpenAI gateway key
    - Start the app locally: Run the following command in the console 

To use my application: 
    - Upload one or more pdf, txt, or md files by browsing local files or dragging and dropping
    - User feedback will begin on the UI after a few seconds as the chatbot starts analysing the documents
    - When the chatbot tells you it is ready, ask question in the chat box

Features of my application include: 
    - Supports multiple file suploads in multiple formats (text files are processed as a whole and PDFs are parsed page-by-page)
    - Splits documents into smaller chunks using the RecursiveCharacterTextSplitter method to maintain some overlap to preserve context
    - Create vector embeddings using Chroma and OpenAI's text-embedding-3-large model for efficient similarity searches
    - Uses LangChain's ConversationalRetrievalChain to allow for conversational question-answers with memory
    - Answers with citations from the text as needed (expandable list including file name and page number)
    - Leverages a fallback system to a safe "stuffed-context" mode where the text is sent to the model to generate an answer directly if RAG is unable to provide a relevant answer
    - Provides guardrails for scanned or image-based PDFs that displays a warning explaining the issue

