import json
from dotenv import load_dotenv

from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder

load_dotenv()


def create_documents(articleJSON):
    with open(articleJSON, "r") as f:
        articles = json.load(f)

    documents = []
    for id in articles:
        article = articles.get(id)
        doc = Document(
            content=article["abstract"],
            meta={
                "pmid": id,
                # "authors": article["authors"],
                "title": article["title"],
                # "publication_year": article["publication_year"],
            },
        )
        documents.append(doc)
    return documents


def setup_AI(articles_path):
    documents = create_documents(articles_path)
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)

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

    retriever = InMemoryBM25Retriever(document_store=document_store)
    prompt_builder = PromptBuilder(template=prompt_template)
    llm = OpenAIGenerator()

    global rag_pipeline
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", llm)
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")


def ask_AI(question):
    results = rag_pipeline.run(
        {
            "retriever": {"query": question, "top_k": 15},
            "prompt_builder": {"question": question},
        }
    )
    response = results["llm"]["replies"][0]
    return response


def main():
    articleJSON = "articles.json"
    setup_AI(articleJSON)

    while True:
        question = input("question: ")
        if question.strip() == "end":
            break
        answer = ask_AI( question)
        print(answer)       


if __name__ == "__main__":
    main()
