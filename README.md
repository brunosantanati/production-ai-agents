## Course
Source code of the course "Production AI Agents with LangChain + LangGraph [2026]". Link: https://www.udemy.com/course/production-ai-agents/

## Code
Original source code (links):
1. The main source code (all projects, except the final production) can be found [here](https://github.com/pdichone/production-course-main-code).
2. The production API project can be found [here](https://github.com/pdichone/lang-production-api).

## Useful links
- [Claude Dashboard](https://platform.claude.com/dashboard)
- [OpenAI Dashboard](https://platform.openai.com/home)
- [LangSmith Dashboard](https://smith.langchain.com/)
- [LangChain Messages](https://docs.langchain.com/oss/python/langchain/messages)

## Commands used
```
uv init
uv venv
source .venv/bin/activate
deactivate
uv add langchain langchain-core langgraph langchain-openai langchain-anthropic python-dotenv
uv add langchain_community
uv add bs4
uv add pypdf
uv add langchain_ollama
uv add sentence-transformers
uv add langchain-huggingface
uv add langchain_chroma
uv add rank_bm25
uv add langgraph-checkpoint-sqlite
uv add pytest
uv sync
touch .env
uv run main.py

API
uv init
uv add langchain-anthropic langgraph langsmith fastapi uvicorn slowapi pydantic-settings python-dotenv
uv add --dev pytest httpx
touch app/__init__.py tests/__init__.py
touch app/config.py app/models.py app/security.py app/cache.py app/monitoring.py app/agent.py app/main.py
touch tests/test_security.py tests/test_cache.py tests/test_api.py
touch .env.example
uv add langchain-openai
./Production-test-commands.sh

Error:
-------
$ uv run advanced_rag.py
snap-confine is packaged without necessary permissions and cannot continue
required permitted capability cap_dac_override not found in current capabilities:
  = 
-------
Fix:
-------
sudo systemctl restart snapd.apparmor apparmor
-------
```
