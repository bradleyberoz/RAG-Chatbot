import json

from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.evaluators import (
    ContextRelevanceEvaluator,
    FaithfulnessEvaluator,
)
from haystack.components.generators import OpenAIGenerator
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

import document_builders
from main import setup_AI

load_dotenv()


def evaluate_pubmedqa(pubmedqa_path, num_questions):
    # different prompt to get yes no maybe answers
    prompt_template = """
    Answer the question based on the provided PubMed abstracts.
    Your answer should be one of: 'yes', 'no', or 'maybe'.

    Also provide the full reasoning for your answer.

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

        # get response
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

        # parse response
        first_line = response.splitlines()[0].lower()
        guessed_answer = "maybe"  # default
        if "yes" in first_line:
            guessed_answer = "yes"
        elif "no" in response:
            guessed_answer = "no"

        if guessed_answer == correct_answer:
            correct += 1

        # evaluate context retrieval
        retrieved_docs = results["retriever"]["documents"]
        contexts = [doc.content for doc in retrieved_docs]
        questions = [question for _ in range(len(contexts))]
        context_evaluator = ContextRelevanceEvaluator()
        context_relevance_result = context_evaluator.run(
            questions=questions, contexts=contexts
        )

        print()
        output = (
            f"Correct answer: {correct_answer}\n"
            f"Model answer: {guessed_answer}\n"
            f"Response: {response}\n"
            f"Context Relevance Score Indiviual: {context_relevance_result["individual_scores"]}\n"
            f"Context Relevance Score Average: {context_relevance_result['score']:.2f}\n"
        )
        print(output)
        outputs.append(output)

    # save output to file
    with open("last_test.txt", "w") as log_file:
        for entry in outputs:
            log_file.write(entry + "\n")

    accuracy = correct / num_questions
    print("Accuracy:", accuracy)
    return accuracy


def eval_custom(question: str):
    # different prompt to get yes no maybe answers
    prompt_template = """
        Answer the question based on the provided PubMed abstracts.
        Your answer should be one of: 'yes', 'no', or 'maybe'.

        Also provide the full reasoning for your answer.

        Context:
        {% for doc in documents %}
            {{ doc.content }}
        {% endfor %}

        Question: {{question}}

        Answer:
        """

    # get response
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

    # parse response
    first_line = response.splitlines()[0].lower()
    guessed_answer = "maybe"  # default
    if "yes" in first_line:
        guessed_answer = "yes"
    elif "no" in response:
        guessed_answer = "no"

    # evaluate context retrieval
    retrieved_docs = results["retriever"]["documents"]
    contexts = [doc.content for doc in retrieved_docs]
    questions = [question for _ in range(len(contexts))]
    context_evaluator = ContextRelevanceEvaluator()
    context_relevance_result = context_evaluator.run(
        questions=questions, contexts=contexts
    )

    print()
    output = (
        f"Model answer: {guessed_answer}\n"
        f"Full Response: {response}\n\n"
        f"Context Relevance Score Indiviual: {context_relevance_result["individual_scores"]}\n"
        f"Context Relevance Score Average: {context_relevance_result['score']:.2f}\n"
        f"Used Document IDs: {[doc.meta["pmid"] for doc in retrieved_docs]}"
    )
    print(output)


if __name__ == "__main__":
    global rag_pipeline
    # pubmedqa_path = "ori_pqal.json"
    # documents = document_builders.build_pubmedqa_documents(pubmedqa_path)
    # evaluate_pubmedqa(pubmedqa_path, 2)

    with open("retrieved_articles.json") as file:
        data = json.load(file)
        for question in data:
            documents = document_builders.qa_docs(data[question])
            rag_pipeline = setup_AI(documents)
            eval_custom(question)
