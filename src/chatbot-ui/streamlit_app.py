import streamlit as st
from openai import OpenAI
from core.config import config
import sys
import os
import requests
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from api.core.config import Settings
from api.rag.retrival import rag_pipeline

client = OpenAI(api_key=config.OPENAI_API_KEY)  # Initialize client using imported API key

def api_call(method, url, **kwargs):

    def show_error_popup(message):
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message
        }
        
    try:
        response = getattr(requests, method)(url, **kwargs)
        
        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {"message": "Invalid response format from server."}
            
        if response.ok:
            return True, response_data
        else:
            return False, response_data
            
    except requests.exceptions.ConnectionError:
        show_error_popup("Connection error. Please try again.")
        return False, {"message": "Connection error. Please try again."}
    except requests.exceptions.Timeout:
        show_error_popup("Request timeout. Please try again later.")
        return False, {"message": "Request timeout. Please try again later."}
    except Exception as e:
        show_error_popup(f"Unexpected error: {str(e)}")
        return False, {"message": f"Unexpected error: {str(e)}"}

if "retrieved_items" not in st.session_state:
    st.session_state.retrieved_items = []

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

if "query_counter" not in st.session_state:
    st.session_state.query_counter = 0

if "sidebar_key" not in st.session_state:
    st.session_state.sidebar_key = 0

if "sidebar_placeholder" not in st.session_state:
    st.session_state.sidebar_placeholder = None

if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# Sidebar - Suggestions
with st.sidebar:
    st.markdown("### Suggestions")
    
    # Create or get the placeholder
    if st.session_state.sidebar_placeholder is None:
        st.session_state.sidebar_placeholder = st.empty()
    
    # Clear and rebuild the suggestions
    with st.session_state.sidebar_placeholder.container():
        if st.session_state.retrieved_items:
            for idx, item in enumerate(st.session_state.retrieved_items):
                st.divider()
                st.caption(item.get('description', 'No description'))
                if 'image_url' in item:
                    st.image(item["image_url"], width=300)
                st.caption(f"Price: {item['price']} USD")
        else:
            st.info("No suggestions yet")

# Main content - Chat interface

# Display all messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Hello! How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.spinner("Thinking..."):
        settings = Settings()  # Create instance
        status, output = api_call("post", f"{settings.API_URL}/rag", json={"query": prompt, "thread_id": st.session_state.session_id})
        # Update retrieved items
        st.session_state.retrieved_items = output.get("used_image_urls", [])
        
        # Clear the sidebar placeholder to force refresh
        if st.session_state.sidebar_placeholder is not None:
            st.session_state.sidebar_placeholder.empty()
        
        response_content = output.get("answer", str(output))
    
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    st.rerun()