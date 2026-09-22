# LangChain + ChromaDB RAG Web Application

This repository contains a complete RAG (Retrieval-Augmented Generation) application built with Flask, LangChain, ChromaDB, and ready for deployment on Render with Gunicorn.

## Project Structure
```text
rag-app/
├── templates/
│   └── index.html      # UI for file upload and chat interface
├── app.py              # Flask backend & RAG pipeline
├── requirements.txt    # Dependencies
├── Procfile            # Deployment instructions for Render
└── README.md           # Documentation
```

## Local Setup & Testing

1. **Clone or unzip project files**:
   ```bash
   cd rag-app
   ```

2. **Create a virtual environment & install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**:
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run Locally**:
   ```bash
   python app.py
   ```
   Open `http://localhost:5000` in your browser.

## Deployment on Render

1. Push this project to a **GitHub** repository.
2. Log in to [Render](https://dashboard.render.com/) and click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. Set the build parameters:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
5. In **Environment Variables**, add:
   - `OPENAI_API_KEY` = `your_actual_openai_api_key`
6. Click **Create Web Service**.
