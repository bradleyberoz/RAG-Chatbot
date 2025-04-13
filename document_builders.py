from haystack import Document

def build_pubmedapi_documents(json_file_path):
    """
    Builds Haystack Document objects from the PubMed API JSON output
    
    Args:
        json_file_path: Path to the JSON file with PubMed article data
        
    Returns:
        List of Haystack Document objects
    """
    import json
    
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        articles_data = json.load(file)
    
    documents = []
    
    # Process the articles data
    for pmid, article_info in articles_data.items():
        # Create a Haystack Document
        doc = Document(
            content=article_info["abstract"],
            meta={
                "pmid": pmid,
                "title": article_info["title"]
            }
        )
        documents.append(doc)
    
    return documents
