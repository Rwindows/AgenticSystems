import streamlit as st
import os
from openai import OpenAI
from core.config import OPENAI_API_KEY

## Lets create a sidebar and list the available models
st.sidebar.title("Chat-UI")
provider = st.sidebar.selectbox("Provider", ["openai"])  # Set provider
model = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4o-2024-08-06", "gpt-4o-2024-08-06", "gpt-4o-2024-08-06", "gpt-4o-2024-08-06"])

#save the provider and model in a session state
st.session_state.provider = provider
st.session_state.model = model

#make dynamic and provide the client ready

client = OpenAI(api_key=OPENAI_API_KEY)  # Initialize client using imported API key

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model=model,
                messages=st.session_state.messages,
            )
            st.write(response.choices[0].message.content)
            st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message.content})