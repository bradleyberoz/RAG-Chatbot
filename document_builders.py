import json
from haystack.dataclasses.document import Document


def build_pubmedqa_documents(path: str):
    """Create a list of documents from pubmedqa dataset

    Args:
        path (str): path to pubmedqa file

    Returns:
        list: list of documents
    """
    with open(path, "r") as f:
        data = json.load(f)

    documents = []
    for id in data:
        curr = data[id]

        doc = Document(
            content="\n".join(curr["CONTEXTS"]),
            meta={
                "pmid": id,
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

        return qa_docs(articles)


def qa_docs(map: dict):
    documents = []
    for id in map:
        article = map.get(id)
        doc = Document(
            content=article["abstract"],
            meta={
                "pmid": id,
                "title": article["title"],
            },
        )
        documents.append(doc)
    return documents
