# RAG Security Lab

Learn how RAG pipelines work by building and breaking them. A hands-on guide for security professionals who want to understand AI systems from the inside out before testing them.

Build the same RAG pipeline in both **LangChain** and **Haystack**, then red-team it with attack scripts mapped to the [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/).

## Quick Start

```bash
# Install Ollama for local models
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# Setup
python -m venv venv && source venv/bin/activate
pip install langchain langchain-community langchain-ollama haystack-ai chromadb pypdf sentence-transformers ollama

# Generate sample corpus, then follow the guide
python create_sample_data.py
```

Everything runs locally — no API keys needed, full visibility into every byte of data.

## What's Covered

The guide walks through five phases, each built in both frameworks side-by-side:

1. **Document Ingestion** — loading, chunking, and preprocessing
2. **Embedding & Storage** — vectorizing chunks into ChromaDB
3. **Retrieval** — similarity search and ranking
4. **Generation** — full RAG chain with prompt templates
5. **Red Teaming** — document poisoning, context stuffing, retrieval manipulation, metadata injection

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) (local LLM runtime)
- ~8 GB disk space for models
- GPU optional

## Resources

- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [LangChain Docs](https://python.langchain.com/docs) · [Haystack Docs](https://docs.haystack.deepset.ai) · [ChromaDB Docs](https://docs.trychroma.com)
- [Garak (LLM vulnerability scanner)](https://github.com/leondz/garak) · [PyRIT (Microsoft AI red teaming)](https://github.com/Azure/PyRIT)

## License

MIT

## Disclaimer

For authorized security testing and educational purposes only. Only test systems you own or have explicit permission to test.
