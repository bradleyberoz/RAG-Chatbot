import json
from data_acquisition import RAG_QueryUser, RAG_ProcessInputToSearchable, RAG_RetrievePubMedArticles, RAG_RetrieveArticleDetails
from rag import setup_AI, ask_AI, build_pubmedapi_documents

def integrated_medical_rag():
    """
    Integrates the Medical Chatbot's data retrieval with the RAG model
    """
    # Step 1: Run the Medical Chatbot to generate the dataset
    print("Generating medical article dataset from PubMed...")
    
    # Get user input
    user_input = RAG_QueryUser()
    
    # Process input into a searchable query
    searchable_query = RAG_ProcessInputToSearchable(user_input)
    print(f"Searchable Query: {searchable_query}")
    
    # Retrieve articles from PubMed
    pmids = RAG_RetrievePubMedArticles(searchable_query)
    
    # Fetch detailed information for each PMID
    articles_info = {}
    for pmid in pmids:
        article_info = RAG_RetrieveArticleDetails(pmid)
        articles_info.update(article_info)
    
    # Save to JSON file that your RAG model can read
    articles_json_path = "articles.json"
    with open(articles_json_path, 'w') as json_file:
        json.dump(articles_info, json_file, indent=4)
    
    print(f"Articles saved to {articles_json_path}")
    
    # Step 2: Set up the RAG pipeline with the generated dataset
    documents = build_pubmedapi_documents(articles_json_path)
    rag_pipeline = setup_AI(documents)
    
    # Step 3: Interactive Q&A using the RAG model
    print("\nRAG model ready. Ask medical questions or type 'end' to exit.")
    while True:
        question = input("Question: ")
        if question.strip().lower() == "end":
            break
        answer = ask_AI(question)
        print(f"\nAnswer: {answer}\n")


def main():
    print("Medical Research Assistant")
    print("1. Use existing article dataset")
    print("2. Generate new dataset from PubMed")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        article_json = input("Enter path to articles JSON file (default: articles.json): ") or "articles.json"
        setup_AI(build_pubmedapi_documents(article_json))
        
        while True:
            question = input("Question: ")
            if question.strip() == "end":
                break
            answer = ask_AI(question)
            print(answer)
    
    elif choice == "2":
        integrated_medical_rag()
    
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
