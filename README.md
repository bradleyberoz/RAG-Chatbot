# RAG-Chatbot
Biomedical chatbot that answers research questions using a RAG framework built with Haystack. 
---

## Modules Overview

### 1. **User Input Handler**
  - Collects user input in the form of a medical question.
  - Returns the processed user query or an error message.

### 2. **Query Processor**
  - Uses OpenAI's GPT-3.5-turbo to convert natural language medical questions into concise, PubMed-optimized search queries.
  - Handles errors during the query transformation process.

### 3. **PubMed Retrieval System**
  - Queries the PubMed API using the processed query and retrieves relevant PubMed IDs (PMIDs).
  - Supports configurable limits on the number of results returned.
  - Fetches detailed article information (title, abstract) for each PMID using the `metapub` library.
  - Returns structured JSON data for easy export.

### 4. **Data Exporter**
- Outputs retrieved article data to a JSON file named `articles.json`.
- Ensures structured, human-readable formatting for further use.

---

## RAG Framework Implementation

This project implements a 4-stage Retrieval-Augmented Generation (RAG) pipeline based on industry standards:

| Stage         | Component                  | Implementation Details                |
|---------------|----------------------------|---------------------------------------|
| **Indexing**    | Vector Database Setup      | PubMed API as the knowledge source    |
| **Retrieval**   | Document Search            | PubMed ID (PMID) lookup system        |
| **Augmentation**| Context Enrichment         | OpenAI-processed queries + PMID data  |
| **Generation**  | Response Synthesis        | Structured JSON output with metadata  |

### Key Features:
- Dynamic updating through PubMed API integration.
- Context-aware query processing using OpenAI's GPT model.
- Source attribution via PMID-based references for transparency.

---

## Setup Instructions

### 1. Install Dependencies
Install all required Python libraries:
- python-dotenv
- openai
- requests
- metapub

### 2. Configure API Keys
Create a `.env` file in the root directory and add your OpenAI API key

### 3. Run the Application
Run the script using Python: 
- python main.py



# WARNING
- Current main branch implementation DOES NOT contain environment variable for OpenAI API key. **The environment variable must be set locally!**

