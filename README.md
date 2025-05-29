# PDF Query Chatbot

A web application that allows users to query PDF documents using different AI models (GPT-4 and Gemini). Built with FastAPI and LlamaIndex.

## Features

- Upload and query multiple PDF documents
- Support for multiple AI models (GPT-4 and Gemini)
- Real-time chat interface
- Document list view
- Model selection

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd pdf-query-chatbot
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

5. Place your PDF files in the `data` directory.

## Usage

Run the application:
```bash
python app.py
```

The application will start at `http://localhost:8000`. You can then:
1. See your PDF documents listed on the left
2. Select your preferred AI model
3. Ask questions about your PDF content
4. Get AI-generated responses

## API Endpoints

- `GET /`: Main web interface
- `POST /query`: Query endpoint
  - Request body: `{"query": "your question", "model": "model_id"}`
  - Response: `{"response": "answer from the model"}`

## Requirements

- Python 3.8+
- OpenAI API key
- Google API key
- PDF files to query

## License

MIT License 