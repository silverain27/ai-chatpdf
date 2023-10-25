from dotenv import load_dotenv
load_dotenv()
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
import tempfile
import os

# 제목 
st.title("Nepes ChatPDF")
st.write("----")

#파일 업로드 
uploaded_file = st.file_uploader("Choose a file")
st.write("----")

def pdf_to_document(uploaded_file):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

#업로드 완성되면 동작 
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # #Loader
    # loader = PyPDFLoader("unsu.pdf")
    # pages = loader.load_and_split()

    #Split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 300, #몇글자 단위로 쪼갤건지 
        chunk_overlap  = 29,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    #Enbedding
    embeddings_model = OpenAIEmbeddings()

    # load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)


    # Question 
    st.header("PDF 에게 질문 해보세요 ")
    question  = st.text_input('질문을 입력하세요 ')
    
    if st.button('질문하기'):
        

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())

        result = qa_chain({"query":question})
        st.write(result)

# docs = retriever_from_llm.get_relevant_documents(query=question)
# print(len(docs))
# print(docs)

