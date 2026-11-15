# Lumen Travel — Firebase Authentication + RAG Chatbot

**Student:** Nguyễn Minh Quân  
**Student ID:** 25127124

## Overview and features

Lumen Travel is a Streamlit travel assistant for Da Nang, Vietnam. It combines Firebase Email/Password Authentication with retrieval-augmented generation (RAG) over a local JSON travel guide.

- Account registration, sign-in, session expiry handling, and sign-out.
- Chat available after sign-in, with history stored in the Streamlit session.
- Answers generated from retrieved guide excerpts, with entry citations.
- Expandable retrieved sources and cosine similarity scores.
- A citation guard replaces answers with an unknown response when citations are missing or refer to entries outside the retrieved set. Valid citations do not guarantee factual support.

## System architecture

```text
Streamlit interface <-> Firebase Authentication (Email/Password REST API)
        |
        | signed-in question
        v
JSON Knowledge Base -> EmbeddingGemma -> cached document embeddings
Question ------------> EmbeddingGemma -> query embedding
                                            |
                               Cosine Similarity (Top-k = 3)
                                            |
                                   Retrieved guide excerpts
                                            |
                              Qwen3 (qwen3:4b-instruct)
                                            |
                              Citation guard -> Streamlit answer
```

The knowledge base is `knowledge_base.json`. Ollama serves both `embeddinggemma` and `qwen3:4b-instruct`, locally or through a Pinggy tunnel. Firebase handles authentication only; the application does not use Firestore.

## Installation and running

Prerequisites: Python 3.11+, a Firebase project with Email/Password Authentication enabled, and an Ollama server with the model names used by `rag.py`.

From the project directory, run these commands in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure `.env` as described below. On the Ollama server, prepare the models:

```text
ollama pull embeddinggemma
ollama pull qwen3:4b-instruct
```

Keep the Ollama server running, then start the interface:

```powershell
streamlit run app.py
```

Open the URL shown by Streamlit. Create an account, sign in, and ask a question about Da Nang. The guide is embedded on the first chat request for each cached guide/server configuration.

## Environment configuration

The committed `.env.example` contains placeholders only:

```dotenv
FIREBASE_API_KEY=your_firebase_web_api_key
OLLAMA_URL=http://localhost:11434
```

1. In Firebase Console, enable **Authentication > Sign-in method > Email/Password**.
2. Copy the project's Web API key from **Project settings > General** into the local `.env` file.
3. Set `OLLAMA_URL` to the local Ollama address or your server's HTTPS Pinggy tunnel URL, without a trailing slash.

Keep actual API keys, passwords, tokens, and tunnel URLs out of the repository. The `.env` file, virtual environment, and Python/test caches are ignored.

## Existing test cases

`tests/test_guide.py` contains three offline tests:

| Test | Purpose |
| --- | --- |
| `test_guide_schema` | Checks guide entry count, unique IDs, and nonempty categories and tags. |
| `test_reject_duplicate_id` | Checks that duplicate guide IDs raise `RAGError`. |
| `test_unknown_citation_is_blocked` | Mocks retrieval and generation to check that an unknown citation triggers the unknown response. |

These are structural and citation-guard tests, not Firebase or Ollama integration tests. **No tests were run during this submission preparation; no pass/fail results are claimed.** To run them separately, install `pytest` and use `python -m pytest -q`.

The existing manual scenarios are retained below. Their outcomes remain unverified.

| Type | Question | Expected entry ID(s) | Result |
| --- | --- | --- | --- |
| Answerable | What is the adult ticket to Marble Mountains in 2026? | G05 | Not tested |
| Answerable | What can I buy at Han Market? | G10 | Not tested |
| Answerable | At what time is the Dragon Bridge show on Friday? | G12 | Not tested |
| Paraphrase | How much do grown-ups pay to visit Thuy Son Peak? | G05 | Not tested |
| Not in guide | How much is a taxi from the airport to My Khe Beach today? | None | Not tested |
| Not in guide | Which hotel has a room available tomorrow? | None | Not tested |

## Submission files

```text
app.py
firebase_client.py
rag.py
knowledge_base.json
requirements.txt
README.md
.env.example
.gitignore
tests/test_guide.py
```

## Demo video

[Watch the Lumen Travel demo on Google Drive](https://drive.google.com/file/d/1Q846DVCzQobD82uhHhks3dOxylYjPasn/view?usp=sharing)

The video is hosted on Google Drive and is not included in this repository.

## Limitations and attribution

The guide covers selected Da Nang topics; reference prices and hours may change. The application requires Firebase and an available Ollama server. Generation may still produce unsupported statements despite the prompt and citation guard.

The original scaffold and seed text were prepared with AI assistance. This submission preparation updates documentation and packaging only; application logic, models, knowledge-base data, and test source are unchanged.
