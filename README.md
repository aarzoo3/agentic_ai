# agentic_ai
Agentic AI is a repository for creating multimodal agents and multi-agent workflows using the Phidata framework. These agents are designed to handle various tasks by leveraging predefined tools, advanced models, and efficient workflows.


# 🧠 Retrieval-Augmented Generation (RAG) Projects

This repository contains two distinct RAG (Retrieval-Augmented Generation) implementations showcasing different approaches to combining large language models with external tools, vector databases, and multi-agent systems.

---

## 🚀 1. Crew AI – Agentic RAG System

An agent-based RAG system built using [CrewAI](https://docs.crewai.com/), designed for modularity and real-world adaptability. It uses multiple specialized agents to process user queries intelligently.

### 🔧 Features

- **Modular Multi-Agent Architecture**:
  - 📜 **Query Rewriting Agent**: Interprets user intent and reformulates the query if needed.
  - 🌐 **Internet Agent**: Uses the [SerperDevTool] tool to fetch real-time web data.
  - 📄 **PDF Agent**: Reads, chunks, and embeds PDF documents for semantic retrieval.
  - 🧠 **Decision-Maker Agent**: Chooses the right agent(s) to handle the current user query.

- **RAG Workflow**:
  - User input is evaluated by the Decision Maker.
  - Based on the query, relevant agents are activated.
  - Retrieved context is passed to a language model for generation.

- ⚡ **FastAPI Integration**:
  - All agents are wrapped in a clean RESTful API for easy deployment and testing.

---

## 🧪 2. LangChain RAG

A traditional RAG pipeline built using [LangChain](https://docs.langchain.com/), designed for simplicity and performance on local data.

### 🔧 Features

- **Embedding with Hugging Face Transformers**:
  - Supports models like `sentence-transformers/all-MiniLM-L6-v2` for semantic understanding.

- **Document Chunking**:
  - Uses `RecursiveCharacterTextSplitter` for optimal chunk sizes and context preservation.

- **Vector Store**:
  - 🔍 Powered by **ChromaDB**, enabling fast and efficient similarity search over text chunks.

- **Pipeline Overview**:
  - Load documents → Chunk → Embed → Store in vector DB → Retrieve based on query → Generate response using LLM

---