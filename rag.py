import json

from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.evaluators import (
    ContextRelevanceEvaluator,
    DocumentMAPEvaluator,
    FaithfulnessEvaluator,
)
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.document_stores.in_memory import InMemoryDocumentStore
from document_builders import build_pubmedapi_documents

load_dotenv()


def setup_AI(documents):
    """
    Sets up a RAG (Retrieval-Augmented Generation) pipeline

    Args:
        documents: List of documents to be indexed and used for retrieval

    Returns:
        A configured RAG pipeline ready for question answering
    """

    # prep and write document store
    document_embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    document_embedder.warm_up()

    documnets_with_embeddings = document_embedder.run(documents)

    document_store = InMemoryDocumentStore()
    document_store.write_documents(documnets_with_embeddings["documents"])

    # prompt template
    prompt_template = """
    You are an expert scientific research assistant. Your answers must be:
    - Factually accurate based solely on the provided context
    - Detailed and comprehensive
    - Include citations to the source documents
    
    If the context doesn't contain enough information to answer properly, say "I don't have enough information to answer this question definitively."
    
    Context:
    {% for doc in documents %}
    ---
    Document ID: {{ doc.meta.pmid }}
    Title: {{ doc.meta.title }}
    
    Content:
    {{ doc.content }}
    {% endfor %}
            
    Question: {{question}}
    Answer:
    """

    text_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    retriever = InMemoryEmbeddingRetriever(document_store)
    prompt_builder = PromptBuilder(template=prompt_template)
    llm = OpenAIGenerator()

    # build pipeline
    global rag_pipeline
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", llm)

    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever", "prompt_builder")
    rag_pipeline.connect("prompt_builder.prompt", "llm")
    return rag_pipeline


def ask_AI(question):
    results = rag_pipeline.run(
        {
            "text_embedder": {"text": question},
            "prompt_builder": {"question": question},
        },
        include_outputs_from=["retriever"],
    )

    response = results["llm"]["replies"][0]

    retrieved_docs = results["retriever"]["documents"]
    contexts = [doc.content for doc in retrieved_docs]
    context_evaluator = ContextRelevanceEvaluator()
    context_relevance_result = context_evaluator.run(
        questions=[question for _ in range(len(contexts))], contexts=contexts
    )
    print(f"Context Relevance Score: {context_relevance_result['score']:.2f}")

    return response


def main():
    articleJSON = "articles.json"
    setup_AI(build_pubmedapi_documents(articleJSON))

    while True:
        question = input("question: ")
        if question.strip() == "end":
            break
        answer = ask_AI(question)
        print(answer)


if __name__ == "__main__":
    main()