###############################################
##########        DESCRIPTION        ##########
###############################################
#
# This script implements a Medical Chatbot that:
#     - Takes a medical question from the user
#     - Converts it to a searchable query using OpenAI's GPT model
#     - Retrieves relevant medical research articles from PubMed API
#     - Returns a list of PMIDs (PubMed IDs) for these articles

##############################################
######     PREPROCESSOR DIRECTIVES     #######
##############################################
import json
import requests
import os                       # used to access environment variable for openai
from dotenv import load_dotenv  # used to access environment variable for openai
from openai import OpenAI       # LLM processing
from metapub import PubMedFetcher
import xml.etree.ElementTree as ET

##############################################
#########      GLOBAL VARIBLES      ##########
##############################################
errStatus = False                      # Error status variable

# Load openai key into project
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # Retrieve environment variable

client = OpenAI(api_key=api_key)       # Initialize OpenAI client

##############################################
#########    FUNCTION DEFINITIONS   ##########
##############################################
"""
Description:  Queries the user to enter input for a medical question
Inputs:       None
Ouputs:       None
Returns       user_input   - User input
"""
def RAG_QueryUser():
    # Prompt the user for input
    user_input = input("Please enter your medical question: ").strip()
        
    # Check if the input is valid (non-empty)
    if not user_input:
        errStatus = True
        return "Invalid Input: Please enter a question."
        
    # Return user input
    return user_input

"""
Description: Processes user input into a 'searchable' format using OpenAI's LLM
Input:       String user_input       - The user's input question.
Output:      None
Returns:     String searchable_query - Searchable query string for research articles
"""
def RAG_ProcessInputToSearchable(user_input):
    # Attempt to process input into a 'searchable query'
    try:
        if (errStatus == True): # Do not proceed if the error status is not false
            return "Invalid Input: Please enter a question."
        else:
            # openai API structure to query LLM
            response = client.chat.completions.create(
            messages = 
            [
                { "role": "user", 
                  "content": f"Convert the following medical question into a concise, searchable query for research articles: {user_input}"
                }
             ],
             model="gpt-3.5-turbo",  # Specify the updated model
            )

        # Extract the generated text from the response
        searchable_query = response.choices[0].message.content
        return searchable_query
        
    except Exception as e:
        print(f"Error processing input with OpenAI: {e}")
        return None

"""
Description:  Retrieves PMIDs using PubMedAPI based on a searchable quer and prints them
Inputs:       searchable_query     - Searchable query string for research articles
              max_results          - Number of articles to retrieve
Ouputs:       None
Returns       pmids                - A .json of PMIDs
"""
def RAG_RetrievePubMedArticles(searchable_query, max_results=20):
    # PubMedAPI endpoint
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    # Set parameters to search for PMIDs with PubMedAPI
    search_params = {
        "db": "pubmed",
        "term": searchable_query,
        "retmode": "json",
        "retmax": max_results
    }

    # Execute search
    search_response = requests.get(base_url, params=search_params)
    
    # Check that the search was successful
    if search_response.status_code != 200:
        print(f"Failed to fetch PMIDs. HTTP Status Code: {search_response.status_code}")
        return []

    # Input search data into a .json
    search_data = search_response.json()
    pmids = search_data.get("esearchresult", {}).get("idlist", [])

    if not pmids: # No PMIDs found
        print("No articles found for the given query.")
        return []

    # Return a list of PMIDs
    return pmids

""" 
Description:  Fetches and parses PubMed article information based on the PMID.
Inputs:       pmid      - The PubMed ID of the article.
Ouputs:       None
Returns       dict      - A dictionary containing the PMID as the key and the title, MeSH terms, and abstract as values
"""
def RAG_RetrieveArticleDetails(pmid):
    fetch = PubMedFetcher()
    try:
        article = fetch.article_by_pmid(pmid)
        
        # Extract Title
        title = article.title
        
        # Extract Abstract
        abstract_text = article.abstract
        
        # Create JSON response
        response = {
            str(pmid): {
                "title": title,
                "abstract": abstract_text
            }
        }
        
        return response
    
    except Exception as e:
        print(f"Error fetching article for PMID {pmid}: {e}")
        return {}

def ErrorHandler():
    # Potential error handling module?
    print("Error")

def create_test_dataset(test_questions):
    """
    Creates a dataset from test questions and their results
    """
    dataset = []
    
    for question in test_questions:
        print(f"Processing test question: {question}")
        
        # Get user input
        user_input = question
        
        # Process input into searchable query
        searchable_query = RAG_ProcessInputToSearchable(user_input)
        
        # Retrieve and process articles
        results = RAG_RetrievePubMedArticles(user_input, searchable_query)
        
        # Validate results
        validation = RAG_ValidateResults(user_input, results["relevant_articles"])
        
        # Add to dataset
        dataset.append({
            "question": question,
            "searchable_query": searchable_query,
            "all_articles": results["all_articles"],
            "relevant_articles": results["relevant_articles"],
            "validation": validation
        })
        
        # Save dataset after each question (to prevent data loss)
        with open("medical_chatbot_test_dataset.json", "w") as f:
            json.dump(dataset, f, indent=4)
            
    return dataset

def integrated_medical_rag():
    """
    Integrates the Medical Chatbot's data retrieval with the RAG model
    """
    # Step 1: Run the Medical Chatbot to generate the dataset
    print("Generating medical article dataset from PubMed...")
    
    # Instead of running as a separate process, import and call directly
    from medical_chatbot import RAG_QueryUser, RAG_ProcessInputToSearchable, RAG_RetrievePubMedArticles, RAG_RetrieveArticleDetails
    
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


"""
Description:  Main method 
Inputs:       None
Ouputs:       None
Returns       None
"""
def main():
    print("Welcome to the Medical Chatbot!")
    """
    # Query User Input
    user_input = RAG_QueryUser()
    
    # Check that user input is valid
    if errStatus:
        print("Error: Invalid input. Please restart and enter a valid question.")
        return
    
    print(f"User Input: {user_input}")
    """

    """
    Processes multiple questions from a file and retrieves PubMed articles for each
    Uses question mark as delimiter to separate questions
    """

    file_path = r"C:\Users\bradl\source\repos\RAG-Chatbot\questions.txt"

    # Read the input file
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Split by question mark to get individual questions
    raw_questions = content.split('?')
    
    # Clean up the questions and remove empty ones
    questions = []
    for q in raw_questions:
        q = q.strip()
        if q:  # Only add non-empty questions
            questions.append(q + '?')  # Add back the question mark
    
    print(f"Found {len(questions)} questions to process")

    all_articles_info = {}  # Dictionary to store all articles' information

    for i in questions:
        # Process input into a 'searchable query' 
        # This is what will be inputted into PubMedAPI to search for research articles
        searchable_query = RAG_ProcessInputToSearchable(i)
        print(f"Searchable Query: {searchable_query}")
    
        # Retrieve articles from PubMed
        pmids = RAG_RetrievePubMedArticles(searchable_query)
    
        print("\nRetrieved Articles:")
    
        # Fetch detailed information for each PMID
        articles_info = {}
        for pmid in pmids:
            article_info = RAG_RetrieveArticleDetails(pmid)
            articles_info.update(article_info)
    
        # Pretty-print the articles into json objects
        print(json.dumps(articles_info, indent=4))

        # Add current question's articles info to the main dictionary
        all_articles_info[i] = articles_info
    
    # Save all articles' information to a JSON file
    output_file_path = "retrieved_articles.json"
    with open(output_file_path, "w") as json_file:
        json.dump(all_articles_info, json_file, indent=4)
    
    print(f"Articles information has been saved to {output_file_path}")
    

    with open('articles.json', 'w') as json_file:
      json.dump(articles_info, json_file, indent=4)

# Run the main function
if __name__ == "__main__":
    main()
