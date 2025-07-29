# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy only necessary files first
COPY pyproject.toml README.md ./
COPY src/chatbot-ui/core ./src/chatbot-ui/core
COPY src/chatbot-ui/streamlit_app.py ./src/chatbot-ui/
COPY src/chatbot-ui/Makefile ./src/chatbot-ui/
COPY src/chatbot-ui/retrival.py ./src/chatbot-ui/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir qdrant-client

# Expose port for Streamlit
EXPOSE 8501

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "src/chatbot-ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]