##############################################
#########          IMPORTS          ##########
##############################################
import json
import requests
import os                       # used to access environment variable for openai
from dotenv import load_dotenv  # used to access environment variable for openai
from openai import OpenAI       # LLM processing
import xml.etree.ElementTree as ET

###############################################
#########          DESCRIPTION        #########
###############################################
#
# This script implements a Medical Chatbot that:
#     - Takes a medical question from the user
#     - Converts it to a searchable query using OpenAI's GPT model
#     - Retrieves relevant medical research articles from PubMed API
#     - Returns a list of PMIDs (PubMed IDs) for these articles
#
###############################################

##############################################
#########      GLOBAL VARIBLES      ##########
##############################################
errStatus = False                             # Error status variable

# Load openai key into project
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # Retrieve environment variable

client = OpenAI(api_key=api_key) #Initialize OpenAI client

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
def RAG_RetrievePubMedArticles(searchable_query, max_results=10):
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

def ErrorHandler():
    # Potential error handling module?
    print("Error")

# Main function to test the workflow
def main():
    print("Welcome to the Medical Chatbot!")
    
    # Query User Input
    user_input = RAG_QueryUser()
    
    # Check that user input is valid
    if errStatus:
        print("Error: Invalid input. Please restart and enter a valid question.")
        return
    
    print(f"User Input: {user_input}")
    
    # Process input into a 'searchable query' 
    # This is what will be inputted into PubMedAPI to search for research articles
    searchable_query = RAG_ProcessInputToSearchable(user_input)
    
    print(f"Searchable Query: {searchable_query}")
    
    # Retrieve articles from PubMed
    articles = RAG_RetrievePubMedArticles(searchable_query)
    
    print("\nRetrieved Articles:")
    
    # Pretty-print the articles into json objects
    print(json.dumps(articles, indent=2))

# Run the main function
if __name__ == "__main__":
    main()