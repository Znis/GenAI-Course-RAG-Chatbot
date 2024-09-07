# Create an OpenAI client.
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st

OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
LANGCHAIN_API_KEY = st.secrets['LANGCHAIN_API_KEY']
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']

LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_ENDPOINT = "https://api.langchain.plus"
LANGCHAIN_PROJECT = "rag-assignment-prod"

llm = ChatOpenAI(  
    openai_api_key=OPENAI_API_KEY,  
    model_name='gpt-3.5-turbo',  
    temperature=0.0  
)

index_name = "rag-assignment"
model_name = "text-embedding-ada-002"

embeddings = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=PINECONE_API_KEY, text_key='nepal-constitution-2072')
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key='answer'
)
retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    chain_type="stuff", 
    return_source_documents=True,
    return_generated_question=True,
)