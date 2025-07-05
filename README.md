# AgenticSystems

Streamlit-based Chat UI for AgenticSystems

## Overview

This project provides a Streamlit-based user interface to interact with the AgenticSystems chatbot. It uses OpenAI's GPT models for natural language responses and python-dotenv to manage environment variables.

## Features

- Real-time chat interface powered by Streamlit
- Integration with OpenAI API
- Automatic code reloading with Watchdog
- Containerized deployment via Docker

## Prerequisites

- Python 3.8 or higher
- [OpenAI API Key](https://platform.openai.com/)
- Docker (optional)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Rwindows/AgenticSystems.git
   cd AgenticSystems
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

4. Create a `.env` file in the project root with your OpenAI API key:
   ```dotenv
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### Run Locally

From the project root, run:

```bash
streamlit run src/chatbot-ui/streamlit_app.py
```

Or use the Makefile:

```bash
make -C src/chatbot-ui run-streamlit
```

Then open your browser at `http://localhost:8501`.

### Docker

Build the Docker image:

```bash
docker build -t agentic-systems-chatbot-ui .
```

Run the container:

```bash
docker run -p 8501:8501 agentic-systems-chatbot-ui
```

## Configuration

- Edit `src/chatbot-ui/core/config.py` to adjust application settings.
- Environment variables are loaded from the `.env` file.

## Contributing

Contributions welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License.