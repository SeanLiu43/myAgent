# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A collection of Python scripts exploring LangChain/LangGraph agent patterns with the Anthropic Claude API. Each script is a standalone chatbot demonstrating a different level of agent complexity. The project language (comments, prompts) is Chinese.

## Scripts

- **generanlChat.py** — Basic conversational chatbot using LangChain's `ChatPromptTemplate` with message history (no tools)
- **ToolsChart.py** — Tool-calling agent built manually with LangChain: defines `search` and `calculator` tools, binds them to the LLM, and handles the tool-call → tool-result → final-answer loop explicitly
- **langGraphChart.py** — Same tools but uses LangGraph's `create_react_agent` for automatic tool-call cycling (graph-based state machine)

## Running

```bash
# Each script is run independently
python generanlChat.py
python ToolsChart.py
python langGraphChart.py
```

Requires a `.env` file with `ANTHROPIC_API_KEY` set.

## Dependencies

- `langchain-anthropic` — Claude LLM integration
- `langchain-core` — prompts, messages, tools
- `langgraph` — graph-based agent (`create_react_agent`)
- `python-dotenv` — loads `.env`

Install with: `pip install langchain-anthropic langchain-core langgraph python-dotenv`

## Key Patterns

- All scripts use `claude-sonnet-4-20250514` as the model
- Tools are defined with the `@tool` decorator from `langchain_core.tools`
- Chat history is managed manually via a `history` list of `HumanMessage`/`AIMessage` objects
- The interactive loop exits on "quit" or "exit"
