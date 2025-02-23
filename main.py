import json
import requests

def get_pubmed_ids(query, max_results=100):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    return data['esearchresult']['idlist']

##
# Function that parses a JSON file and extracts the PMID and final_decision for each entry
# Inputs: file_path - the path to the JSON file
#         num_entries - the number of entries to process
# Returns: a 2D list containing the PMID and final_decision for each entry
##
def parse_json_file(file_path, num_entries):
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Create a 2D list to store the results
    results = []
    
    # Iterate through the specified number of entries
    for i, (pmid, entry) in enumerate(data.items()):
        if i >= num_entries:
            break
        
        # Extract PMID and final_decision
        final_decision = entry.get('final_decision', '')
        results.append([pmid, final_decision])
    
    return results

file_path = 'C:\\Users\\bradl\\OneDrive\\Desktop\\New folder\\ori_pqaa.json'
num_entries = int(input("Enter the number of entries to process: "))

parsed_data = parse_json_file(file_path, num_entries)

# Print the results
print("PMID\t\tFinal Decision")
print("--------------------------")
for pmid, decision in parsed_data:
    print(f"{pmid}\t{decision}")