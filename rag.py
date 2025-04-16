from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.evaluators import (
    ContextRelevanceEvaluator,
    FaithfulnessEvaluator,
)
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from document_builders import build_pubmedapi_documents

load_dotenv()


def setup_AI(documents: list[Document]):
    """
    Sets up a RAG (Retrieval-Augmented Generation) pipeline

    Args:
        documents: List of documents to be indexed and used for retrieval

    Returns:
        RAG pipeline ready for question answering
    """

    # prep and write document store
    document_embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    document_embedder.warm_up()

    documents_with_embeddings = document_embedder.run(documents)

    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents_with_embeddings["documents"])

    # prompt templates
    global prompt_templates
    prompt_templates = {
        "open": """
            You are an expert scientific research assistant. Your answers must be:
            - Factually accurate based solely on the provided context
            - Detailed and comprehensive
            - Include citations to the source documents
            
            If the context doesn't contain enough information to answer properly, say "I don't have enough information to answer this question definitively."
            
            Context:
            {% for doc in documents %}
            ---
            Document ID: {{ doc.meta.pmid }}
            Title: {{ doc.meta.title }}
            
            Content:
            {{ doc.content }}
            {% endfor %}
                    
            Question: {{question}}
            Answer:
            """,
        "yes_no": """
            Answer the question based on the provided PubMed abstracts.
            Your answer should be one of: 'yes', 'no', or 'maybe'.

            Also provide the full reasoning for your answer.

            Context:
            {% for doc in documents %}
                {{ doc.content }}
            {% endfor %}

            Question: {{question}}

            Answer:
            """,
    }

    text_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    retriever = InMemoryEmbeddingRetriever(document_store)
    prompt_builder = PromptBuilder(template=prompt_templates["open"])
    llm = OpenAIGenerator()

    # build pipeline
    global rag_pipeline
    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", llm)

    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever", "prompt_builder")
    rag_pipeline.connect("prompt_builder.prompt", "llm")
    return rag_pipeline


def ask_AI(question: str, question_type="open") -> str:
    """
    Ask a question to RAG

    Args:
        question: The user question.
        question_type: 'open' or 'yes_no'.

    Returns:
        The LLM-generated answer string.
    """
    results = run_pipeline(question, question_type)

    response = results["llm"]["replies"][0]

    return response


def run_pipeline(question: str, question_type="open"):
    """
    Run the RAG pipeline with a specific prompt template.

    Args:
        question: The input question.
        question_type: Type of question ('open' or 'yes_no')

    Returns:
        The generated result dict.
    """

    template = prompt_templates[question_type]
    return rag_pipeline.run(
        {
            "text_embedder": {"text": question},
            "prompt_builder": {"question": question, "template": template},
        },
        include_outputs_from=["retriever"],
    )


def eval_AI(question: str, question_type="open", correct_answer=None):
    """
    Run evaluation for a given question.

    Includes:
    - Context relevance
    - Faithfulness

    Args:
        question: The question being asked.
        question_type: Type of question ('open' or 'yes_no')
        correct_answer: (optional) ground truth answer
    Returns:
        Dictionary containing evaluation results.


        "question":
            The original input question.

        "correct_answer":
            The answer if provided; otherwise, None.

        "yes_no_answer":
            'yes', 'no', or 'maybe' if question_type is "yes_no".

        "full_response":
            The full answer generated by the LLM.

        "context_relevance":
            "individual":
                List of relevance scores (0 to 1) for each document used.

            "average":
                Average relevance score for all documents.

        "faithfulness":
            "score":
                A score (0 to 1) representing how well the answer aligns with the retrieved documents.

        "documents_used":
            A list of document PMIDs that were used to generate the answer.

    """
    result = run_pipeline(question, question_type)

    response = result["llm"]["replies"][0]

    if question_type == "yes_no":
        first_line = response.splitlines()[0].lower()
        yes_no_answer = "maybe"
        if "yes" in first_line:
            yes_no_answer = "yes"
        elif "no" in first_line:
            yes_no_answer = "no"
    else:
        yes_no_answer = None

    # Retrieval evaluation
    retrieved_docs = result["retriever"]["documents"]
    contexts = [doc.content for doc in retrieved_docs]
    questions = [question] * len(contexts)

    context_result = ContextRelevanceEvaluator().run(
        questions=questions, contexts=contexts
    )
    faithfulness_result = FaithfulnessEvaluator().run(
        questions=questions,
        predicted_answers=[response] * len(contexts),
        contexts=contexts,
    )

    return {
        "question": question,
        "correct_answer": correct_answer,
        "yes_no_answer": yes_no_answer,
        "full_response": response,
        "context_relevance": {
            "individual": context_result["individual_scores"],
            "average": context_result["score"],
        },
        "faithfulness": {"score": faithfulness_result["score"]},
        "documents_used": [doc.meta["pmid"] for doc in retrieved_docs],
    }


def pretty_format_evaluation(result: dict) -> str:
    """
    Formats the evaluation result into a string.
    """
    output = [
        f"Question: {result['question']}",
    ]

    if result.get("correct_answer") is not None:
        output.append(f"Correct Answer: {result['correct_answer']}")
    if result.get("yes_no_answer") is not None:
        output.append(f"Yes/No Answer: {result['yes_no_answer']}")

    output.extend(
        [
            f"\nFull Response:\n{result['full_response']}\n",
            f"Context Relevance:",
            f" - Individual Scores: {result['context_relevance']['individual']}",
            f" - Average Score: {result['context_relevance']['average']:.2f}",
            f"Faithfulness Score: {result['faithfulness']['score']:.2f}",
            f"Documents Used: {result['documents_used']}",
        ]
    )

    return "\n".join(output)


def main():
    articleJSON = "articles.json"
    setup_AI(build_pubmedapi_documents(articleJSON))

    while True:
        question = input("question: ")
        if question.strip() == "end":
            break
        answer = ask_AI(question)
        # answer = pretty_format_evaluation(eval_AI(question))
        print(answer)


if __name__ == "__main__":
    main()
