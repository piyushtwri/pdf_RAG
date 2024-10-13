import os


##------------------------------------------------------------------------------------------
def load_pdf(pdf_file_name):
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(pdf_file_name)
    documents = loader.load_and_split()
    return documents
##------------------------------------------------------------------------------------------
def load_pdf_from_dir(pdf_directory):
    from langchain.document_loaders import PyPDFLoader
    documents = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):  # Check if the file is a PDF
            file_path = os.path.join(pdf_directory, filename)
            # Load the PDF using PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())  # Load and append the documents
    # # Now, 'documents' contains all the loaded PDF documents from the directory
    # print(f"Loaded {len(documents)} documents from the PDF files.")
    return documents


##------------------------------------------------------------------------------------------
def split_text(document,chunk_size=600, chunk_overlap=100, verbose=False):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text = text_splitter.split_documents(document)
    if verbose == True:
        print(f"Document of {len(document)} pages split into total {len(text)} chunks of {chunk_size} characters")
    return text
##------------------------------------------------------------------------------------------
def vector_retriever(text, persistDirectory='chromaDB',embedding_model="sentence-transformers/paraphrase-MiniLM-L12-v2", verbose=True):
    from langchain_chroma import Chroma
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    embedding = HuggingFaceEmbeddings(model_name=embedding_model)
    # embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(persistDirectory):
        vector_store= Chroma(persist_directory=persistDirectory, embedding_function=embedding)
        if verbose:
            print(f"Directory {persistDirectory} already exists.")
    else:
        vector_store = Chroma.from_documents(documents=text, embedding=embedding, persist_directory=persistDirectory)
        if verbose:
            print(f"Directory {persistDirectory} created.")

    retriever=vector_store.as_retriever()
    return retriever
##------------------------------------------------------------------------------------------

##------------------------------------------------------------------------------------------
def quiz_untill_stop(chain):
    # Maintain a context string for the conversation
    conversation_context = ""
    chat_history = []  # Initialize chat history as an empty list

    # Initial question
    x = "In one line tell me what is the document's heading or name"
    while (x != "stop") and (x != "exit"):
        # Update context with the current question and the last answer
        full_context = conversation_context + "\nContext:\n" + conversation_context + "\nQuestion:\n" + x

        # Get the answer from the chain
        answer = chain.invoke({'context': full_context, 'question': x, 'chat_history': chat_history})['answer']
        
        print(f"Answer: {answer}")
        
        # Update the conversation context and chat history
        conversation_context += f"Q: {x}\nA: {answer}\n"
        chat_history.append((x, answer))  # Append the current Q&A to the history

        # Get the next question from the user
        x = input("User_question:\n  ")
##------------------------------------------------------------------------------------------

##------------------------------------------------------------------------------------------

##------------------------------------------------------------------------------------------

##------------------------------------------------------------------------------------------

##------------------------------------------------------------------------------------------

##------------------------------------------------------------------------------------------

##------------------------------------------------------------------------------------------

##------------------------------------------------------------------------------------------