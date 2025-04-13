import json
import tkinter as tk
from tkinter import scrolledtext, messagebox
from data_acquisition import RAG_ProcessInputToSearchable, RAG_RetrievePubMedArticles, RAG_RetrieveArticleDetails
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

        # Show processing message
        self.answer_area.insert(tk.END, f"Processing your question: {question}\n")
        self.answer_area.see(tk.END)
        self.root.update()  # Update the UI to show the processing message

        # Generate dataset and set up RAG pipeline if not already done
        if not self.dataset_ready:
            self.answer_area.insert(tk.END, "Generating dataset for the first time. This may take a moment...\n")
            self.answer_area.see(tk.END)
            self.root.update()
            
            self.generate_dataset_and_setup_pipeline(question)
            
            if not self.dataset_ready:
                return  # Exit if dataset generation failed

        # Get the answer from the RAG model
        try:
            answer = ask_AI(question)
            self.answer_area.insert(tk.END, f"\nAnswer: {answer}\n\n")
            self.answer_area.see(tk.END)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get an answer: {e}")
            self.answer_area.insert(tk.END, f"Error: {str(e)}\n\n")
            self.answer_area.see(tk.END)

    def generate_dataset_and_setup_pipeline(self, user_input):
        try:
            # Step 1: Process the user input directly from the UI
            self.answer_area.insert(tk.END, "Processing your query...\n")
            self.root.update()
            
            searchable_query = RAG_ProcessInputToSearchable(user_input)
            
            self.answer_area.insert(tk.END, "Retrieving relevant medical articles...\n")
            self.root.update()
            
            pmids = RAG_RetrievePubMedArticles(searchable_query)
            
            if not pmids:
                self.answer_area.insert(tk.END, "No relevant articles found. Please try a different query.\n\n")
                return
                
            self.answer_area.insert(tk.END, f"Found {len(pmids)} relevant articles. Retrieving details...\n")
            self.root.update()

            articles_info = {}
            for i, pmid in enumerate(pmids):
                self.answer_area.insert(tk.END, f"Processing article {i+1}/{len(pmids)}...\r")
                self.root.update()
                article_info = RAG_RetrieveArticleDetails(pmid)
                articles_info.update(article_info)

            articles_json_path = "articles.json"
            with open(articles_json_path, 'w') as json_file:
                json.dump(articles_info, json_file, indent=4)

            self.answer_area.insert(tk.END, "Setting up AI model with retrieved data...\n")
            self.root.update()
            
            # Step 2: Set up the RAG pipeline
            documents = build_pubmedapi_documents(articles_json_path)
            self.rag_pipeline = setup_AI(documents)
            self.dataset_ready = True  # Mark the dataset as ready
            
            self.answer_area.insert(tk.END, "Dataset ready! Processing your question now...\n")
            self.root.update()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate dataset and set up pipeline: {e}")
            self.answer_area.insert(tk.END, f"Error generating dataset: {str(e)}\n\n")
            self.answer_area.see(tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalResearchAssistantApp(root)
    root.mainloop()
