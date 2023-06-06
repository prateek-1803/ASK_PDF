# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import streamlit as st
from htmlTemplate import css1,bot_template,user_template
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from Summary import split_text

# Gets text from PDF
def get_text(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    text = text.replace('\t',' ')
    return text

# Splits obtained text into chunks
def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks

# Creates embeddings for chunks and stores them in a vectorstore

def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts = chunks , embedding=embeddings)
    return vectorstore

# Creates a conversation chain between User and AI

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature = 0)
    memory = ConversationBufferMemory(memory_key = 'chat_history' , return_messages = True)
    conversation = ConversationalRetrievalChain.from_llm(llm = llm, retriever=vectorstore.as_retriever(),
                                                         memory = memory)
    return conversation

# Uses the conversation chain to answer the given question
def handle_question(question):
    response = st.session_state.conversation({'question' : question})
    st.session_state.chat_history = response['chat_history']
    for i,message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}" , message.content) , unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def get_summary(text,summary):
    summary += split_text(text)
    st.success(summary)
def main():
    load_dotenv()
    st.set_page_config(page_title="CHAT WITH MULTIPLE PDFS!",page_icon=":books:")
    st.write(css1,unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat Here :)")
    summary = "Summary : "
    # Input question
    question = st.text_input("Ask a question about your pdf:")
    if question:
        handle_question(question)

    # The templates for User and AI responses
    st.write(user_template.replace("{{MSG}}" , "Hello AI!") , unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Human! How may I help you?"), unsafe_allow_html=True)

    with st.sidebar:
        # Uploading Pdfs
        st.subheader("Your Pdfs")
        pdfs = st.file_uploader("Upload your pdfs (Only Pdfs) here and click on 'Upload'" , accept_multiple_files=True)

        if st.button("Upload"):
            with st.spinner("Uploading.."):
                text = get_text(pdfs)
                chunks = get_text_chunks(text)
                vectorstore = get_vectorstore(chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
        st.subheader("NOTE : Summarizing text for a large pdf will take more time and provide a broader summary"
                     ", Nevertheless you will get a rough idea")
        if st.button("Summarize"):
            with st.spinner('This may take a couple of minutes..'):
                source_text = get_text(pdfs)
                get_summary(source_text,summary)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/