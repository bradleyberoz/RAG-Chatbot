import json

from dotenv import load_dotenv

import document_builders
import rag

load_dotenv()


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

        output = rag.eval_AI(
            question,
            question_type="yes_no",
            correct_answer=correct_answer,
        )

        if output.get("yes_no_answer") == correct_answer:
            correct += 1

        print()

        print(rag.pretty_format_evaluation(output))
        outputs.append(output)

    # save output to file
    with open("last_test.txt", "w") as log_file:
        for entry in outputs:
            log_file.write(entry + "\n")

    accuracy = correct / num_questions
    print("Accuracy:", accuracy)
    return accuracy


def eval_custom(question: str):
    print(rag.pretty_format_evaluation(rag.eval_AI(question, "yes_no")))


if __name__ == "__main__":
    global rag_pipeline
    # pubmedqa_path = "ori_pqal.json"
    # documents = document_builders.build_pubmedqa_documents(pubmedqa_path)
    # rag_pipeline = rag.setup_AI(documents)
    # evaluate_pubmedqa(pubmedqa_path, 2)

    with open("retrieved_articles.json") as file:
        data = json.load(file)
        for i, question in enumerate(data):
            if i == 1:
                break
            documents = document_builders.qa_docs(data[question])
            rag_pipeline = rag.setup_AI(documents)
            eval_custom(question)
