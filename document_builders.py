import json
from haystack.dataclasses.document import Document


def build_pubmedqa_documents(path):
    with open(path, "r") as f:
        data = json.load(f)

    documents = []
    for id in data:
        curr = data[id]

        doc = Document(
            content="\n".join(curr["CONTEXTS"]),
            meta={
                "question": curr["QUESTION"],
                "answer": curr["final_decision"],
                "year": curr["YEAR"],
            },
        )
        documents.append(doc)
    return documents


def build_pubmedapi_documents(path):
    with open(path, "r") as f:
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
