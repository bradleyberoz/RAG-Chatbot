import json
from dotenv import load_dotenv
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.evaluators import ContextRelevanceEvaluator
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.rankers import TransformersSimilarityRanker
from main import setup_AI
from document_builders import build_pubmedqa_documents

load_dotenv()


def evaluate_pubmedqa(pubmedqa_path, num_questions):

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

    with open(pubmedqa_path, "r") as f:
        data = json.load(f)

    correct = 0
    outputs = []

    for i, id in enumerate(data):
        if i >= num_questions:
            break

        question = data[id]["QUESTION"]
        correct_answer = data[id]["final_decision"]

        results = rag_pipeline.run(
            {
                "text_embedder": {"text": question},
                "prompt_builder": {
                    "question": question,
                    "template": prompt_template,
                },
            },
            include_outputs_from=["retriever"],
        )
        response = results["llm"]["replies"][0]
        first_line = response.splitlines()[0].lower()

        predicted_answer = "maybe"  # default
        if "yes" in first_line:
            predicted_answer = "yes"
        elif "no" in response:
            predicted_answer = "no"

        if predicted_answer == correct_answer:
            correct += 1

        retrieved_docs = results["retriever"]["documents"]
        contexts = [doc.content for doc in retrieved_docs]
        context_evaluator = ContextRelevanceEvaluator()
        context_relevance_result = context_evaluator.run(
            questions=[question for _ in range(len(contexts))], contexts=contexts
        )
        print()

        output = (
            f"Question {i+1}: {question}\n"
            f"Correct answer: {correct_answer}\n"
            f"Model answer: {predicted_answer}\n"
            f"Response: {response}\n"
            # f"Context Relevance Score Indiviual: {[context_relevance_result['score']:.2f]}"
            f"Context Relevance Score Average: {context_relevance_result['score']:.2f}"
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
    global rag_pipeline
    rag_pipeline = setup_AI(build_pubmedqa_documents(pubmedqa_path))
    evaluate_pubmedqa(pubmedqa_path, 20)
