import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# Show title and description.
st.title("💬 Nepal Constitution 2072 Chatbot")
st.write(
    "This is a conversational chatbot where you can ask "
    "questions regarding the Constitution of Nepal 2072."
)
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
LANGCHAIN_API_KEY = st.secrets['LANGCHAIN_API_KEY']
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']

LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_ENDPOINT = "https://api.langchain.plus"
LANGCHAIN_PROJECT = "rag-assignment-prod"


def handle_streaming_json(streaming_json):    
    for chunk in streaming_json:
        if 'answer' in chunk:
            yield chunk['answer'] 

# Create an OpenAI client.
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

# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.
if len(st.session_state) <= 0:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("Ask a question"):

    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate a response using the OpenAI API.
    output = qa.stream({'question': prompt})
    # Stream the response to the chat using `st.write_stream`, then store it in 
    # session state.
    with st.chat_message("assistant"):
        response = st.write_stream(handle_streaming_json(output))
    st.session_state.messages.append({"role": "assistant", "content": response})
