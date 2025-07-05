# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir streamlit openai python-dotenv watchdog

# Expose port for Streamlit
EXPOSE 8501

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "src/chatbot-ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]