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
uv sync
touch .env
uv run main.py
```
