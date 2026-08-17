<!-- -----------------------  Header  ----------------------------- -->
<h1 align="center">Production Grade RAG System</h1>

<p align="center">

  <img alt="profile views" src="https://komarev.com/ghpvc/?username=PrabhakaranVijay&label=Profile%20Views&color=E10600&style=flat"/>
  
  <img alt="Github top language" src="https://img.shields.io/github/languages/top/PrabhakaranVijay/ELIZA?color=E10600">

  <img alt="Github language count" src="https://img.shields.io/github/languages/count/PrabhakaranVijay/Prod-RAG?color=E10600">

  <img alt="Repository size" src="https://img.shields.io/github/repo-size/PrabhakaranVijay/Prod-RAG?color=E10600">

  <img alt="License" src="https://img.shields.io/github/license/PrabhakaranVijay/Prod-RAG?color=E10600">

  <img alt="Github issues" src="https://img.shields.io/github/issues/PrabhakaranVijay/Prod-RAG?color=E10600" />

  <img alt="Github forks" src="https://img.shields.io/github/forks/PrabhakaranVijay/Prod-RAG?color=E10600" />

  <img alt="Github stars" src="https://img.shields.io/github/stars/PrabhakaranVijay/Prod-RAG?color=E10600" />
</p>

<h4 align="center"> 
	🚧  Service Sphere 🚀 Under construction...  🚧
</h4>

<hr>
<!-- ======================= Navigation ======================= -->

<p align="center">
<a href="#project-overview">Project Overview</a> |
<a href="#features">Features</a> |
<a href="#technologies">Technologies</a> |
<a href="#requirements">Requirements</a> |
<a href="#starting">Getting Started</a> |
<a href="#license">License</a> |
<a href="https://github.com/PrabhakaranVijay">Author</a>
</p>

<br>

<!-- ======================= Project Overview ======================= -->

## 🎯 Project Overview

<!-- **ELIZA** - Named after the world's first chatbot, ELIZA is my personal AI home assistant designed to unify voice interaction, knowledge retrieval, automation, and intelligent decision-making across my digital and physical environment. Inspired by JARVIS, ELIZA connects calendars, email, notes, home automation, local infrastructure, and AI models into a single conversational interface. -->

---
enterprise-grade-multimodel-rag/
│
├── app/
│   │
│   ├── main.py
│   │
│   ├── config/
│   │   ├── settings.py
│   │   ├── logging.py
│   │   └── constants.py
│   │
│   ├── llms/
│   │   ├── provider.py
│   │   ├── groq.py
│   │   ├── huggingface.py
│   │   ├── ollama.py
│   │   └── router.py
│   │
│   ├── embeddings/
│   │   ├── embedding_factory.py
│   │   └── bge_embeddings.py
│   │
│   ├── vectorstore/
│   │   ├── pgvector.py
│   │   ├── chroma.py
│   │   └── qdrant.py
│   │
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   ├── metadata.py
│   │   └── pipeline.py
│   │
│   ├── retrieval/
│   │   ├── hybrid_search.py
│   │   ├── bm25.py
│   │   ├── vector_search.py
│   │   └── retriever.py
│   │
│   ├── reranking/
│   │   ├── bge_reranker.py
│   │   └── reranker.py
│   │
│   ├── chains/
│   │   ├── rag_chain.py
│   │   ├── query_rewrite.py
│   │   ├── answer_generation.py
│   │   └── citation_chain.py
│   │
│   ├── evaluation/
│   │   ├── ragas_eval.py
│   │   ├── retrieval_eval.py
│   │   └── llm_eval.py
│   │
│   ├── observability/
│   │   ├── langfuse_client.py
│   │   ├── mlflow_client.py
│   │   └── tracing.py
│   │
│   ├── api/
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   └── dependencies.py
│   │
│   └── utils/
│       ├── helpers.py
│       ├── file_utils.py
│       └── tokenizer.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
│
├── experiments/
│   ├── notebooks/
│   └── evaluation_results/
│
├── tests/
│   ├── test_llms.py
│   ├── test_retrieval.py
│   ├── test_reranking.py
│   └── test_rag.py
│
├── .env
├── .env.example
├── requirements.txt
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
├── README.md
└── .gitignore

---

## 🚀 Getting Started

### 1) Install Dependencies

```bash
uv sync
```

### 2) Configure Environment Variables

Create a `.env` file in the project root and copy values from `.env.example`.

Required values:

- `GROQ_API_KEY`
- `LLM_MODEL` (default: `llama-3.3-70b-versatile`)
- `EMBEDDING_MODEL` (default: `BAAI/bge-m3`)
- `RERANKER_MODEL` (default: `BAAI/bge-reranker-v2-m3`)

### 3) Run The App

Ask with default prompt:

```bash
uv run python -m app.main
```

Ask with your own question:

```bash
uv run python -m app.main "What does the handbook say about leave policy?"
```

The app automatically indexes `data/raw/company_handbook.txt` when present and answers using Groq + RAG retrieval.

### 4) Run Tests

```bash
uv run python -m pytest
```

---

## Contributing

Pull requests are welcome. Please adhere to the established `.eslintrc` conventions and use feature branches.

---

## 🔐 Environment Variables

Some security keys and API credentials are not included in the repository.
If you want to run the full project, please contact me and I will provide the required configuration.

---

## 📄 License

This project is under license from MIT. For more details, see the [LICENSE](LICENSE.md) file.

Made with ❤️ by <a href="https://github.com/PrabhakaranVijay" target="_blank">Team PitWall</a>

&#xa0;

---

## ⭐ If You Like This Project

Give it a ⭐ on GitHub and feel free to fork or improve it!

<br/>
<br/>
<a href="#top" align="center">Back to top</a>
