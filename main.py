import json
import tkinter as tk
from tkinter import scrolledtext, messagebox
from data_acquisition import RAG_QueryUser, RAG_ProcessInputToSearchable, RAG_RetrievePubMedArticles, RAG_RetrieveArticleDetails
from rag import setup_AI, ask_AI, build_pubmedapi_documents

class MedicalResearchAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Research Assistant")
        self.root.geometry("600x400")

        # Input field for questions
        self.question_label = tk.Label(root, text="Ask a medical question:")
        self.question_label.pack(pady=5)

        self.question_entry = tk.Entry(root, width=80)
        self.question_entry.pack(pady=5)

        # Button to submit the question
        self.ask_button = tk.Button(root, text="Ask", command=self.ask_question)
        self.ask_button.pack(pady=5)

        # Scrolled text area for displaying answers
        self.answer_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=15)
        self.answer_area.pack(pady=10)

        # Initialize RAG pipeline
        self.rag_pipeline = None
        self.dataset_ready = False  # Flag to check if the dataset is ready

    def ask_question(self):
        question = self.question_entry.get().strip()
        if not question:
            messagebox.showwarning("Input Error", "Please enter a question.")
            return

        # Generate dataset and set up RAG pipeline if not already done
        if not self.dataset_ready:
            self.generate_dataset_and_setup_pipeline()

        if not self.rag_pipeline:
            messagebox.showerror("Error", "Failed to set up the RAG pipeline.")
            return

        # Get the answer from the RAG model
        try:
            answer = ask_AI(question)
            self.answer_area.insert(tk.END, f"Q: {question}\nA: {answer}\n\n")
            self.answer_area.see(tk.END)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get an answer: {e}")

    def generate_dataset_and_setup_pipeline(self):
        try:
            # Step 1: Run the Medical Chatbot to generate the dataset
            user_input = RAG_QueryUser()
            searchable_query = RAG_ProcessInputToSearchable(user_input)
            pmids = RAG_RetrievePubMedArticles(searchable_query)

            articles_info = {}
            for pmid in pmids:
                article_info = RAG_RetrieveArticleDetails(pmid)
                articles_info.update(article_info)

            articles_json_path = "articles.json"
            with open(articles_json_path, 'w') as json_file:
                json.dump(articles_info, json_file, indent=4)

            # Step 2: Set up the RAG pipeline
            documents = build_pubmedapi_documents(articles_json_path)
            self.rag_pipeline = setup_AI(documents)
            self.dataset_ready = True  # Mark the dataset as ready
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate dataset and set up pipeline: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalResearchAssistantApp(root)
    root.mainloop()