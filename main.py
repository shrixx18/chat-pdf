from fastapi import FastAPI, Body, File, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
 
# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define functions for processing (previously used in Streamlit)

def get_pdf_text(pdf_file: UploadFile):
    """Extracts text from uploaded PDF file"""
    text = ""
    with pdf_file.file as f:
        pdf_reader = PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text



def get_text_chunks(text):
    """Splits text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Creates a vector store for efficient search"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Consider in-memory storage for performance

# def generate_summary_map_reduce(pdf_file: UploadFile):
#     """Generates a summary of a large PDF using MapReduce with GenAI and vector embeddings."""
#     text = get_pdf_text(pdf_file)
#     text_chunks = get_text_chunks(text)

#     # Vector embeddings
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index2")
#     # LLM for summarization
#     llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.5)  # Replace with your model

#     # Map step prompt
#     map_template = """The following is a set of documents
#     {docs}
#     Based on this list of docs, please identify the main themes
#     Helpful Answer:"""
#     map_prompt = PromptTemplate.from_template(map_template)
#     map_chain = LLMChain(llm=llm, prompt=map_prompt)

#     # Reduce step prompt
#     reduce_template = """The following is a set of summaries:
#     {docs}
#     Take these and distill it into a final, consolidated summary of the main themes.
#     Helpful Answer:"""
#     reduce_prompt = PromptTemplate.from_template(reduce_template)

#     # Chain creation
#     reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
#     combine_documents_chain = StuffDocumentsChain(
#         llm_chain=reduce_chain, document_variable_name="docs"
#     )
#     reduce_documents_chain = ReduceDocumentsChain(
#         combine_documents_chain=combine_documents_chain,
#         collapse_documents_chain=combine_documents_chain,
#         token_max=4000,  # Adjust as needed
#     )
#     map_reduce_chain = MapReduceDocumentsChain(
#         llm_chain=map_chain,
#         # map_chain=map_chain,
#         reduce_documents_chain=reduce_documents_chain,
#         document_variable_name="docs",
#         return_intermediate_steps=False,  # Set to True for debugging
#     )
#     text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#         chunk_size=1000, chunk_overlap=0
#     )
#     split_docs = text_splitter.split_documents(text)

#     # Generate summary
#     summary = map_reduce_chain.run(split_docs)

#     return summary

def get_conversational_chain():
    """Defines the question answering chain"""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    if the answer is not available in the context just say, "answer is not available in the context",don't provide the wrong answer\n\n
    Context: \n {context}?\n
    Question: \n{question}\n
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain



def answer_question(question: str, pdf_file: UploadFile = File(...)):
    """Processes PDF, searches for answers, and returns response"""
    # Extract text from PDF
    text = get_pdf_text(pdf_file)
    # Split text into chunks
    text_chunks = get_text_chunks(text)
    # Create vector store (consider caching for performance)
    get_vector_store(text_chunks)

    # Load pre-built conversational chain
    chain = get_conversational_chain()

    # Search for similar documents and answer the question
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(question)
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)

    return response["output_text"]





# Create FastAPI application
app = FastAPI()

allowed_origins = ["http://localhost:3000"]


# Define API endpoint to answer questions from uploaded PDFs
@app.get('/')
async def root():
    return {"message": "Hello from FastAPI!"}
# Define allowed origins (adjust as needed)


@app.post("/answer_question")
async def answer_from_pdf(question: str = Body(...), pdf_file: UploadFile = File(...)):
    """API endpoint to answer questions from uploaded PDFs"""
    try:
        answer = answer_question(question, pdf_file)
        return {"answer": answer}
    except Exception as e:
    
        print(f"Error processing PDF: {e}")
        return {"error": str(e)}
    


# @app.post("/summary")
# async def generate_summary_from_pdf(pdf_file: UploadFile = File(...)):
#     """API endpoint to generate a summary of an uploaded PDF using MapReduce."""
#     try:
#         summary = generate_summary_map_reduce(pdf_file)
#         return {"summary": summary}
#     except Exception as e:
#         print(f"Error generating summary: {e}")
#         return {"error": str(e)}

from fastapi.middleware.cors import CORSMiddleware


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)  # Change port if needed
