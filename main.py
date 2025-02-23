import json
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI
import xml.etree.ElementTree as ET

load_dotenv()


# Global boolean to track input validity
errStatus = False
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key = OPENAI_API_KEY
)

# Function to query the user and handle errors
def RAG_QueryUser():
    """
    Description: Queries the user for input
    Input:       None
    Output:      None
    Returns:     String user_input - The user's input
            
    """
    # Prompt the user for input
    user_input = input("Please enter your medical question: ").strip()
        
    # Check if the input is valid (non-empty)
    if not user_input:
        errStatus = True
        return "Invalid Input: Please enter a question."
        
    # Return success status and user input
    return user_input

def RAG_ProcessInputToSearchable(user_input):
    """
    Description: Processes user input into a 'searchable' format using OpenAI's LLM
    Input:       String user_input       - The user's input question.
    Output:      None
    Returns:     String searchable_query - Searchable query string for research articles
    """
    try:
        if (errStatus == True): # Do not proceed if the error status is not false
            return "Invalid Input: Please enter a question."
        else:
            # Use OpenAI's chat model to process the input into a searchable format
            response = client.chat.completions.create(
            messages=[
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

def RAG_RetrievePubMedArticles(searchable_query, max_results=10):
    """
    Retrieves PMIDs from PubMed based on a searchable query and prints them.

    Args:
        searchable_query (str): The query string to search PubMed.
        max_results (int): Maximum number of results to retrieve (default is 100).

    Returns:
        list: A list of PMIDs.
    """
    # Base URL for PubMed API endpoint
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    # Search PubMed for PMIDs
    search_params = {
        "db": "pubmed",
        "term": searchable_query,
        "retmode": "json",
        "retmax": max_results
    }

    search_response = requests.get(base_url, params=search_params)
    
    if search_response.status_code != 200:
        print(f"Failed to fetch PMIDs. HTTP Status Code: {search_response.status_code}")
        return []

    search_data = search_response.json()
    pmids = search_data.get("esearchresult", {}).get("idlist", [])

    if not pmids:
        print("No articles found for the given query.")
        return []

    # Print all PMIDs
    print("Retrieved PMIDs:")
    for pmid in pmids:
        print(pmid)

    return pmids

# Main function to test the workflow
def main():
    """
    Main function to test the complete workflow of querying, processing, and retrieving articles.
    """
    print("Welcome to the Medical Chatbot!")
    
    # Step 1: Query User Input
    user_input = RAG_QueryUser()
    
    if errStatus:
        print("Error: Invalid input. Please restart and enter a valid question.")
        return
    
    print(f"User Input: {user_input}")
    
    # Step 2: Process Input into Searchable Query
    searchable_query = RAG_ProcessInputToSearchable(user_input)
    
    print(f"Searchable Query: {searchable_query}")
    
    # Step 3: Retrieve Articles from PubMed (Simulated)
    articles = RAG_RetrievePubMedArticles(searchable_query)
    
    print("\nRetrieved Articles:")
    
    # Pretty-printing the articles JSON object
    print(json.dumps(articles, indent=2))

# Run the main function
if __name__ == "__main__":
    main()
