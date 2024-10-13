import os
import shutil  # to delete the existing directory
from utils import load_pdf,split_text,vector_retriever,quiz_untill_stop
from langchain_groq import ChatGroq

pdf_file_name = "docs/coffee_and_health.pdf"

model_name="sentence-transformers/all-MiniLM-L6-v2"
pages=load_pdf(pdf_file_name)
text= split_text(pages,chunk_size=500,chunk_overlap=100)
retriever=vector_retriever(text, verbose=True)


groq_api_key = os.environ['GROQ_API_KEY']
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=None, timeout=None, max_retries=2)


from langchain.chains import ConversationalRetrievalChain
chain = ConversationalRetrievalChain.from_llm( llm=llm,retriever=retriever)

quiz_untill_stop(chain)