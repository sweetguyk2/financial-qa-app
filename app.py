# 23 Aug 2025 V3 - Streamlit Application File

import streamlit as st
import pandas as pd
import torch
import re
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
    Trainer,
    TrainingArguments)
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_cohere import CohereRerank
import cohere
import numpy as np
from streamlit.errors import StreamlitSecretNotFoundError
from huggingface_hub import HfFolder # Import HfFolder to potentially get token

RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 200
RAG_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RAG_GENERATION_MODEL = "distilgpt2" # Placeholder for RAG generation model

FT_MODEL_GPT2_PATH = "sweetguyk2/gpt2-finetuned-financial-qa" # Your Hugging Face repo ID
FT_MODEL_FLAN_T5_PATH = "sweetguyk2/flan-t5-finetuned-financial-qa" # Your Hugging Face repo ID


FINETUNE_DATA_PATH = "jpmc_finetune.jsonl" # Used for data loading, not model loading in app
FINANCIAL_DATA_PATH = "JPMC_Financials.xlsx" # Ensure this file is in your repo for Streamlit Cloud

HF_TOKEN = None
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except KeyError:
    st.info("Hugging Face token not found in Streamlit secrets. Loading public models only.")
except Exception as e:
    st.error(f"An error occurred while accessing Hugging Face token: {e}")

# --- Data Loading and Processing ---
@st.cache_data
def load_and_process_financial_data(file_path):
    """Loads financial data from Excel and converts it into text documents."""
    # FIX: Check for file existence relative to the app's root for Streamlit Cloud
    if not os.path.exists(file_path):
        st.error(f"Error: Financial data file not found at {file_path}. Please ensure 'JPMC_Financials.xlsx' is in your application's repository.")
        return []

    documents = []
    try:
        xls = pd.ExcelFile(file_path)
        for sheet in xls.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet)
            # Attempt to handle potential non-string column names for 'Item'
            item_col = 'Item' if 'Item' in df.columns else df.columns[0]

            year_cols = [col for col in df.columns if isinstance(col, (int, float))]

            for _, row in df.iterrows():
                item = row[item_col]
                if pd.isna(item):
                    continue
                for year in year_cols:
                    value = row[year]

                    try:
                         formatted_value = f"${value:,}" if pd.notna(value) else "N/A"
                    except TypeError:

                         formatted_value = str(value) if pd.notna(value) else "N/A"

                    sentence = f"{sheet} - Value for {item} in year {int(year)} is {formatted_value}"
                    documents.append(Document(page_content=sentence))
        st.success(f"Successfully loaded and processed data from {file_path}")
        return documents
    except FileNotFoundError:
        # This should be caught by the initial check, but keeping as fallback
        st.error(f"Error: Financial data file not found at {file_path}.")
        return []
    except Exception as e:
        st.error(f"An error occurred while processing the financial data: {e}")
        return []

@st.cache_data
def chunk_documents(_documents, chunk_size, chunk_overlap):
    """Splits documents into chunks."""
    if not _documents:
        st.warning("No documents provided for chunking.")
        return []
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(_documents)
        processed_chunks = []
        source_id = "JPMC2024"
        size = chunk_size
        for i, chunk in enumerate(chunks):
            chunk_id = f"{source_id}_size{size}_chunk{i+1}"
            metadata = {
                "source_id": source_id,
                "chunk_size": size,
                "chunk_index": i + 1,
                **chunk.metadata
            }
            # Langchain's Document object expects page_content and metadata
            processed_chunks.append(Document(page_content=chunk.page_content, metadata={"chunk_id": chunk_id, **metadata}))
        st.success(f"Successfully chunked documents into {len(processed_chunks)} chunks.")
        return processed_chunks
    except Exception as e:
        st.error(f"An error occurred while chunking documents: {e}")
        return []

# --- RAG Components ---
@st.cache_data
def setup_rag_retrievers(_documents, _chunks, embedding_model_name):
    """Sets up the Chroma vector store, BM25 retriever, and Cohere re-ranker."""
    if not _documents or not _chunks:
        st.warning("Documents or chunks not available for setting up RAG retrievers.")
        return None, None, None

    # Set your Cohere API key from environment variables or Streamlit secrets
    COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
    if not COHERE_API_KEY:
        try:
            COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
        except StreamlitSecretNotFoundError:
            st.error("Cohere API key not found in environment variables or Streamlit secrets. Reranking will be skipped.")
            COHERE_API_KEY = None
        except KeyError:
            st.error("Cohere API key not found in Streamlit secrets. Reranking will be skipped.")
            COHERE_API_KEY = None


    try:
        # Dense Retriever (ChromaDB)
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        vector_store = Chroma.from_documents(
            documents=_chunks,
            embedding=embeddings
        )
        dense_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        st.success("Successfully set up Dense Retriever (ChromaDB).")

        # Sparse Retriever (BM25)
        sparse_retriever = BM25Retriever.from_documents(_documents)
        sparse_retriever.k = 5
        st.success("Successfully set up Sparse Retriever (BM25).")

        # Re-ranker (Cohere)
        reranker = None
        if COHERE_API_KEY:
            try:
                reranker = CohereRerank(
                    cohere_api_key=COHERE_API_KEY,
                    model="rerank-english-v3.0",
                    top_n=3
                )
                st.success("Successfully set up Cohere Re-ranker.")
            except Exception as e:
                st.error(f"An error occurred while setting up Cohere Re-ranker: {e}")
                reranker = None
        else:
            st.info("Cohere API key not available. Skipping Cohere Re-ranker setup.")

        return dense_retriever, sparse_retriever, reranker
    except Exception as e:
        st.error(f"An error occurred while setting up RAG components: {e}")
        return None, None, None


def hybrid_retrieval(query, dense_retriever, sparse_retriever, reranker):
    """Performs hybrid retrieval and re-ranking."""
    if not dense_retriever or not sparse_retriever:
        st.warning("Retrievers not available for hybrid retrieval.")
        return []

    try:
        # Stage 1: Broad Retrieval
        dense_results = dense_retriever.invoke(query)
        sparse_results = sparse_retriever.invoke(query)

        # Combine and deduplicate
        candidate_docs_dict = {doc.page_content: doc for doc in dense_results + sparse_results}
        candidate_docs = list(candidate_docs_dict.values())

        if not candidate_docs:
            st.info("No candidate documents retrieved from dense or sparse retrieval.")
            return []

        # Stage 2: Re-ranking (if reranker is available)
        if reranker:
            try:
                reranked_docs = reranker.compress_documents(
                    documents=candidate_docs,
                    query=query,
                )
                st.success(f"Successfully reranked {len(candidate_docs)} documents to top {len(reranked_docs)}.")
                return reranked_docs
            except Exception as e:
                st.error(f"An error occurred during Cohere re-ranking: {e}")

                return candidate_docs
        else:
            st.info("Cohere Re-ranker not available. Skipping re-ranking.")
            return candidate_docs

    except Exception as e:
        st.error(f"An error occurred during hybrid retrieval: {e}")
        return []


@st.cache_data
def setup_rag_llm(model_id, token=None):
    """Sets up the generative LLM for RAG."""
    try:
        # FIX: Pass token to from_pretrained
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        model = AutoModelForCausalLM.from_pretrained(model_id, token=token)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        text_generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.95,
            do_sample=True
        )

        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        st.success(f"Successfully loaded RAG Generative LLM: {model_id}")
        return llm, tokenizer
    except Exception as e:
        st.error(f"An error occurred while loading the RAG generative model {model_id}: {e}")
        return None, None


def generate_rag_answer(query, retrieved_chunks, llm):
    """Generates an answer using the RAG model based on retrieved context."""
    if not llm or not retrieved_chunks:
        st.warning("LLM or retrieved chunks not available for RAG answer generation.")
        return "Could not generate an answer."

    context = "\n\n".join([doc.page_content for doc in retrieved_chunks])

    template = """
    Answer the question based only on the following context:

    {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(template)

    try:
        formatted_prompt = prompt.format(context=context, question=query)
        final_response = llm.invoke(formatted_prompt)
        response_text = final_response.strip()
        answer_prefix = "Answer:"
        if answer_prefix in response_text:
            response_text = response_text.rsplit(answer_prefix, 1)[-1].strip()

        return response_text
    except Exception as e:
        st.error(f"An error occurred during RAG response generation: {e}")
        return "Could not generate an answer."


# --- Fine-Tuned Model Components ---
@st.cache_data
def load_fine_tuned_model(model_path, model_type, token=None):
    """Loads the fine-tuned model and tokenizer from a specified path or Hugging Face Hub ID."""
    try:
        if "flan-t5" in model_type.lower():
             # FIX: Pass token to from_pretrained
             tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
             model = AutoModelForSeq2SeqLM.from_pretrained(model_path, token=token)
        else: # Assuming GPT2 or similar causal language model
            # FIX: Pass token to from_pretrained
            tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
            model = AutoModelForCausalLM.from_pretrained(model_path, token=token)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

        st.success(f"Successfully loaded Fine-tuned model from {model_path}")
        return model, tokenizer
    except Exception as e:
        st.error(f"An error occurred while loading the fine-tuned model from {model_path}: {e}")
        return None, None

def generate_ft_answer(query, model, tokenizer, model_type):
    """Generates an answer using the fine-tuned model."""
    if not model or not tokenizer:
        st.warning(f"Fine-tuned model or tokenizer not available for {model_type} answer generation.")
        return f"Could not generate an answer using the fine-tuned {model_type} model."

    try:
        device = model.device # Get the device the model is on

        if "flan-t5" in model_type.lower():
             input_text = f"Question: {query}"
             inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
             # Move input tensors to the correct device
             inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
             output_tokens = model.generate(inputs["input_ids"], max_length=128)
             generated_answer = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        else: # GPT2
            input_text = f"Question: {query}\nAnswer:"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
            # Move input tensors to the correct device
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            output_tokens = model.generate(
                inputs['input_ids'],
                max_length=128,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                temperature=0.7,
                attention_mask=inputs.get('attention_mask'),
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            generated_answer = generated_text.split("Answer:")[-1].strip()

        return generated_answer
    except Exception as e:
        st.error(f"An error occurred during fine-tuned model inference for {model_type}: {e}")
        return f"Could not generate an answer using the fine-tuned {model_type} model."


# --- Streamlit App ---
st.title("Financial QA System")

st.write("Ask questions about the JPMC financial data.")

# Model Selection
model_choice = st.radio(
    "Choose a model:",
    ('RAG System', 'Fine-Tuned GPT2-Medium', 'Fine-Tuned Flan-T5 Small')
)

# User Input
user_question = st.text_input("Enter your question:")

# Process the question when the user hits Enter or the text input loses focus
if user_question:
    st.write(f"You asked: {user_question}")
    st.write("Generating answer...")

    answer = "Could not generate an answer."

    if model_choice == 'RAG System':
        # Load and process data for RAG
        # Assuming FINANCIAL_DATA_PATH is relative to the app's root in Streamlit Cloud
        documents = load_and_process_financial_data(FINANCIAL_DATA_PATH)
        if documents:
            chunks = chunk_documents(documents, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP)
            # Pass COHERE_API_KEY directly to setup_rag_retrievers
            dense_retriever, sparse_retriever, reranker = setup_rag_retrievers(documents, chunks, RAG_EMBEDDING_MODEL)
            if dense_retriever and sparse_retriever:
                retrieved_chunks = hybrid_retrieval(user_question, dense_retriever, sparse_retriever, reranker)
                if retrieved_chunks:
                    # For the RAG system, we are using the base distilgpt2 model, not a fine-tuned one from a path
                    # FIX: Pass HF_TOKEN to setup_rag_llm
                    rag_llm, rag_tokenizer = setup_rag_llm(RAG_GENERATION_MODEL, token=HF_TOKEN)
                    if rag_llm:
                        answer = generate_rag_answer(user_question, retrieved_chunks, rag_llm)
                    else:
                        answer = "Failed to load the RAG generation model."
                else:
                    answer = "Failed to retrieve relevant information for RAG."
            else:
                 answer = "Failed to set up RAG retrievers."
        else:
            answer = "Failed to load or process financial data."

    elif model_choice == 'Fine-Tuned GPT2-Medium':
         # Load fine-tuned GPT2 model from the saved path
         # FIX: Pass HF_TOKEN to load_fine_tuned_model
         ft_model_gpt2, ft_tokenizer_gpt2 = load_fine_tuned_model(FT_MODEL_GPT2_PATH, "gpt2-medium", token=HF_TOKEN)
         if ft_model_gpt2 and ft_tokenizer_gpt2:
             answer = generate_ft_answer(user_question, ft_model_gpt2, ft_tokenizer_gpt2, "gpt2-medium")
         else:
             answer = "Failed to load the fine-tuned GPT2-Medium model."

    elif model_choice == 'Fine-Tuned Flan-T5 Small':
         # Load fine-tuned Flan-T5 model from the saved path
         # FIX: Pass HF_TOKEN to load_fine_tuned_model
         ft_model_flan_t5, ft_tokenizer_flan_t5 = load_fine_tuned_model(FT_MODEL_FLAN_T5_PATH, "flan-t5-small", token=HF_TOKEN)
         if ft_model_flan_t5 and ft_tokenizer_flan_t5:
             answer = generate_ft_answer(user_question, ft_model_flan_t5, ft_tokenizer_flan_t5, "flan-t5-small")
         else:
              answer = "Failed to load the fine-tuned Flan-T5 Small model."

    st.subheader("Answer:")
    st.write(answer)
