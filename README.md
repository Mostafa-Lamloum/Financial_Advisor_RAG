# Financial Advisor Chatbot

This project is a **Financial Advisor Chatbot** that leverages advanced machine learning techniques to provide accurate and contextually relevant financial advice. The chatbot uses Retrieval-Augmented Generation (RAG) to answer user questions by accessing a knowledge base built from financial documents.

## Project Overview

The chatbot is designed to assist users by answering financial questions based on a corpus of data collected from PDF documents. The project involves the following key steps:

1. **Data Collection**:
   - Financial documents in PDF format are collected as the primary data source.
  
2. **Data Processing**:
   - The PDF documents are processed to extract text, which is then divided into smaller, manageable chunks.
  
3. **Embedding Generation**:
   - The text chunks are converted into vector embeddings using a BERT-based model. These embeddings capture the semantic meaning of the text, enabling efficient similarity searches.

4. **Vector Database**:
   - The embeddings are upserted into a Pinecone vector database. Pinecone allows for efficient storage and retrieval of high-dimensional vectors.

5. **Question Answering**:
   - When a user asks a question, the chatbot computes the cosine similarity between the question embedding and the stored embeddings to identify the most similar paragraphs.
   - The relevant paragraphs are retrieved and used as context.

6. **Answer Construction**:
   - With the support of the Gemini model, the chatbot constructs a well-contextualized and informative answer based on the retrieved paragraphs.

## Key Features

- **Retrieval-Augmented Generation (RAG)**: Combines the strengths of retrieval-based systems and generative models to provide accurate and context-aware answers.
- **Semantic Search**: Utilizes BERT embeddings and cosine similarity to find the most relevant information in the knowledge base.
- **Contextual Understanding**: Gemini enhances the chatbot's ability to generate nuanced and contextually appropriate responses.


