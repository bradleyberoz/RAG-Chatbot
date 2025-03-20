import json
from dotenv import load_dotenv
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder

load_dotenv()

def create_documents_from_pubmedqa(pubmedqa_path):
    with open(pubmedqa_path, "r") as f:
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

def setup_pubmedqa_pipeline(pubmedqa_path):
    documents = create_documents_from_pubmedqa(pubmedqa_path)
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)

    prompt_template = """
    Answer the question based on the provided PubMed abstracts.
    Your answer should be one of: 'yes', 'no', or 'maybe'.
    Provide a short explanation for your answer.

    Context:
    {% for doc in documents %}
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

def evaluate_pubmedqa(pubmedqa_path, num_questions):
    with open(pubmedqa_path, "r") as f:
        data = json.load(f)

    correct = 0
    outputs = []
    
    for i, id in enumerate(data):
        if i >= num_questions:
            break


        question = data[id]["QUESTION"]
        correct_answer = data[id]["final_decision"]

        results = rag_pipeline.run({
            "retriever": {"query": question},
            "prompt_builder": {"question": question},
        })

        response = results["llm"]["replies"][0]
        first_line = response.splitlines()[0].lower()
        
        predicted_answer = "maybe"  # default
        if "yes" in first_line:
            predicted_answer = "yes"
        elif "no" in response:
            predicted_answer = "no"

        if predicted_answer == correct_answer:
            correct += 1

        
        output = (
            f"Question {i+1}: {question}\n"
            f"Correct answer: {correct_answer}\n"
            f"Model answer: {predicted_answer}\n"
            f"Response: {response}\n"
        )

        print(output)
        outputs.append(output)

    with open("last_test.txt", "w") as log_file:
        for entry in outputs:
            log_file.write(entry + "\n")
            
        

   
    accuracy = correct / num_questions
    print("Accuracy:", accuracy)
    return accuracy


if __name__ == "__main__":
    pubmedqa_path = "ori_pqal.json"
    setup_pubmedqa_pipeline(pubmedqa_path)
    evaluate_pubmedqa(pubmedqa_path,2)
