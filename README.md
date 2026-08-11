# RAG-Powered Knowledge Assistant 🤖📚

An intelligent, interactive CLI chatbot designed to provide highly accurate, context-driven answers using Retrieval-Augmented Generation (RAG). Built with a focus on reliability and traceability, this agent queries a local vector database to fetch relevant knowledge and strictly answers based on retrieved context while providing source citations. 

## 🚀 Key Features
- **Retrieval-Augmented Generation (RAG):** Combines the reasoning capabilities of large language models with a local knowledge base to prevent hallucinations and provide factual answers.
- **Agentic Workflow:** Utilizes LangChain's agent framework to autonomously determine when to use search tools for fetching context.
- **Strict Sourcing & Citation:** Engineered with custom system prompts that mandate the agent to cite source page references and explicitly report when information is missing from the knowledge base.
- **High-Performance Tech Stack:** 
  - **LangChain:** For agent and tool orchestration.
  - **ChromaDB:** For persistent vector storage and semantic similarity search.
  - **Groq:** Lightning-fast inference using 120B parameter open-source models.
  - **Ollama:** Local, efficient text embeddings using `nomic-embed-text`.

## 🛠️ Built With
* Python
* LangChain
* ChromaDB
* Groq API
* Ollama
