import streamlit as st
from rag_backend import qa, memory
from utils import handle_streaming_json

# Show title and description.
st.title("💬 Nepal Constitution 2072 Chatbot")
st.write(
    "This is a conversational chatbot where you can ask "
    "questions regarding the Constitution of Nepal 2072."
)
# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.
if len(st.session_state) <= 0:
    st.session_state.messages = []
    memory.clear()

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
