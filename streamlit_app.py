import streamlit as st
from data_acquisition import RAG_ProcessInputToSearchable, RAG_RetrievePubMedArticles, RAG_RetrieveArticleDetails
from rag import setup_AI, ask_AI
from haystack import Document

def main():
    st.title("Medical Research Assistant")

    user_input = st.text_input("Enter a medical question:")

    if st.button("Submit"):
        if not user_input.strip():
            st.error("Please enter a valid medical question.")
            return

        # Step 1: Convert to searchable query
        searchable_query = RAG_ProcessInputToSearchable(user_input)
        st.write(f"Searchable Query: {searchable_query}")

        # Step 2: Retrieve articles
        pmids = RAG_RetrievePubMedArticles(searchable_query)
        if not pmids:
            st.warning("No articles found.")
            return

        articles_info = {}
        for pmid in pmids:
            article_info = RAG_RetrieveArticleDetails(pmid)
            articles_info.update(article_info)

        # Step 3: Convert to Haystack documents
        documents = []
        for pmid, data in articles_info.items():
            doc = Document(
                content=data["abstract"],
                meta={"pmid": pmid, "title": data["title"]}
            )
            documents.append(doc)

        # Step 4: Setup and use the RAG pipeline
        rag_pipeline = setup_AI(documents)
        answer = ask_AI(user_input)

        st.subheader("Answer")
        st.write(answer)

if __name__ == "__main__":
    main()
