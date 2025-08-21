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
    TrainingArguments,
)
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
from streamlit.errors import StreamlitSecretNotFoundError # Import the specific exception

# Set your Cohere API key from environment variables or Streamlit secrets
# Ensure you set this in your deployment environment
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
if not COHERE_API_KEY:
    try:
        COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
    except StreamlitSecretNotFoundError:
         # Error message handled by the final check below
         pass
    except KeyError:
         # Handle KeyError specifically for st.secrets access
         pass


if not COHERE_API_KEY:
    st.error("Cohere API key not found. Please set the COHERE_API_KEY environment variable or use Streamlit secrets.")
    st.stop()

# The Cohere API key will be picked up by CohereRerank directly from environment variables or Streamlit secrets
# No need to explicitly set os.environ here.


# --- Global Settings ---
RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 200
RAG_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RAG_GENERATION_MODEL = "distilgpt2" # Using distilgpt2 as in the notebook
FT_MODEL_GPT2 = "gpt2-medium" # Using gpt2-medium as in the notebook
FT_MODEL_FLAN_T5 = "google/flan-t5-small" # Using flan-t5-small as in the notebook
FINETUNE_DATA_PATH = "jpmc_finetune.jsonl" # Assuming this is accessible in the deployed environment or you'll adjust the path
FINANCIAL_DATA_PATH = "JPMC_Financials.xlsx" # Assuming this is accessible in the deployed environment or you'll adjust the path

# --- Data Loading and Processing ---
@st.cache_resource
def load_and_process_financial_data(file_path):
    """Loads financial data from Excel and converts it into text documents."""
    documents = []
    try:
        xls = pd.ExcelFile(file_path)
        for sheet in xls.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet)
            # Attempt to handle potential non-string column names for 'Item'
            item_col = 'Item' if 'Item' in df.columns else df.columns[0] # Assume the first column is the item if 'Item' is not found

            year_cols = [col for col in df.columns if isinstance(col, (int, float))]

            for _, row in df.iterrows():
                item = row[item_col]
                if pd.isna(item): # Skip rows with no item
                    continue
                for year in year_cols:
                    value = row[year]
                    # Format value carefully to avoid errors with different data types
                    try:
                         # Attempt to format as number with commas
                         formatted_value = f"${value:,}" if pd.notna(value) else "N/A"
                    except TypeError:
                         # If formatting fails, just use the raw value
                         formatted_value = str(value) if pd.notna(value) else "N/A"

                    sentence = f"{sheet} - Value for {item} in year {int(year)} is {formatted_value}"
                    documents.append(Document(page_content=sentence))
        return documents
    except FileNotFoundError:
        st.error(f"Error: Financial data file not found at {file_path}. Please ensure the file is in the correct location.")
        return []
    except Exception as e:
        st.error(f"An error occurred while processing the financial data: {e}")
        return []

@st.cache_resource
def chunk_documents(_documents, chunk_size, chunk_overlap):
    """Splits documents into chunks."""
    if not _documents:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(_documents)
    processed_chunks = []
    source_id = "JPMC2024" # Or dynamically generate based on file name/timestamp
    size = chunk_size # Use the actual chunk size used
    for i, chunk in enumerate(chunks):
        chunk_id = f"{source_id}_size{size}_chunk{i+1}"
        metadata = {
            "source_id": source_id,
            "chunk_size": size,
            "chunk_index": i + 1,
            **chunk.metadata # Include original document metadata if any
        }
        # Langchain's Document object expects page_content and metadata
        processed_chunks.append(Document(page_content=chunk.page_content, metadata={"chunk_id": chunk_id, **metadata}))

    return processed_chunks


# --- RAG Components ---
@st.cache_resource
def setup_rag_retrievers(_documents, chunks, embedding_model_name, cohere_api_key):
    """Sets up the Chroma vector store, BM25 retriever, and Cohere re-ranker."""
    if not documents or not chunks:
        return None, None, None

    try:
        # Dense Retriever (ChromaDB)
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        vector_store = Chroma.from_documents(
            documents=chunks, # Use chunks for the vector store
            embedding=embeddings
        )
        dense_retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 for dense

        # Sparse Retriever (BM25)
        # BM25 works best on the original documents or slightly larger chunks
        # Using the initial documents here as they represent complete entries
        sparse_retriever = BM25Retriever.from_documents(documents)
        sparse_retriever.k = 5 # Retrieve top 5 for sparse

        # Re-ranker (Cohere)
        reranker = CohereRerank(
            cohere_api_key=cohere_api_key,
            model="rerank-english-v3.0", # Use a suitable re-ranker model
            top_n=3 # Get top 3 after re-ranking
        )

        return dense_retriever, sparse_retriever, reranker
    except Exception as e:
        st.error(f"An error occurred while setting up RAG components: {e}")
        return None, None, None


def hybrid_retrieval(query, dense_retriever, sparse_retriever, reranker):
    """Performs hybrid retrieval and re-ranking."""
    if not dense_retriever or not sparse_retriever or not reranker:
        return []

    try:
        # Stage 1: Broad Retrieval
        dense_results = dense_retriever.invoke(query)
        sparse_results = sparse_retriever.invoke(query)

        # Combine and deduplicate
        # Using page_content for deduplication
        candidate_docs_dict = {doc.page_content: doc for doc in dense_results + sparse_results}
        candidate_docs = list(candidate_docs_dict.values())

        if not candidate_docs:
            return []

        # Stage 2: Re-ranking
        reranked_docs = reranker.compress_documents(
            documents=candidate_docs,
            query=query,
        )
        return reranked_docs
    except Exception as e:
        st.error(f"An error occurred during hybrid retrieval: {e}")
        return []


@st.cache_resource
def setup_rag_llm(model_id):
    """Sets up the generative LLM for RAG."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        # Set pad_token_id if it's not already set (common for GPT models)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        text_generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256, # Increased max_new_tokens for potentially longer answers
            temperature=0.1,
            top_p=0.95,
            do_sample=True # Enable sampling for more varied output if needed
        )

        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        return llm, tokenizer # Return tokenizer as well if needed later
    except Exception as e:
        st.error(f"An error occurred while loading the RAG generative model: {e}")
        return None, None


def generate_rag_answer(query, retrieved_chunks, llm):
    """Generates an answer using the RAG model based on retrieved context."""
    if not llm or not retrieved_chunks:
        return "Could not generate an answer."

    # Concatenate the content of the retrieved documents
    context = "\n\n".join([doc.page_content for doc in retrieved_chunks])

    # Define the prompt template
    template = """
    Answer the question based only on the following context:

    {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(template)

    try:
        # Format the prompt with the context and question
        formatted_prompt = prompt.format(context=context, question=query)

        # Generate the final answer
        # Use invoke directly with the formatted prompt
        final_response = llm.invoke(formatted_prompt)

        # Post-process the response to remove the prompt template if the model echoes it
        # This is a common issue with causal models and simple prompts
        response_text = final_response.strip()
        answer_prefix = "Answer:"
        if answer_prefix in response_text:
             # Find the last occurrence of "Answer:" and take everything after it
            response_text = response_text.rsplit(answer_prefix, 1)[-1].strip()

        return response_text
    except Exception as e:
        st.error(f"An error occurred during RAG response generation: {e}")
        return "Could not generate an answer."


# --- Fine-Tuned Model Components ---
@st.cache_resource
def load_fine_tuned_model(model_id):
    """Loads the fine-tuned model and tokenizer."""
    try:
        if "flan-t5" in model_id.lower():
             tokenizer = AutoTokenizer.from_pretrained(model_id)
             model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        else: # Assume causal LM like GPT-2
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            # Set pad token for causal models if needed
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token


        return model, tokenizer
    except Exception as e:
        st.error(f"An error occurred while loading the fine-tuned model {model_id}: {e}")
        return None, None

def generate_ft_answer(query, model, tokenizer, model_id):
    """Generates an answer using the fine-tuned model."""
    if not model or not tokenizer:
        return "Could not generate an answer using the fine-tuned model."

    try:
        if "flan-t5" in model_id.lower():
             # Prepare input for Seq2Seq model (Flan-T5)
             input_text = f"Question: {query}"
             inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)

             # Generate output
             output_tokens = model.generate(inputs["input_ids"], max_length=128) # Use greedy search or other generation strategies

             # Decode the output
             generated_answer = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        else: # Assume causal LM (GPT-2)
            # Prepare input for Causal LM
            input_text = f"Question: {query}\nAnswer:"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)

            # Generate output
            output_tokens = model.generate(
                inputs['input_ids'],
                max_length=128, # Control the length of the generated text
                num_return_sequences=1,
                do_sample=True, # Use sampling for more diverse answers
                top_p=0.95,
                top_k=50,
                temperature=0.7,
                 # Explicitly pass attention_mask for GPT-2 with padding
                attention_mask=inputs.get('attention_mask'),
                pad_token_id=tokenizer.eos_token_id # Ensure pad token is set for generation
            )

            # Decode the generated output and extract the answer part
            generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            # Extract the part after "Answer:"
            generated_answer = generated_text.split("Answer:")[-1].strip()


        return generated_answer
    except Exception as e:
        st.error(f"An error occurred during fine-tuned model inference: {e}")
        return "Could not generate an answer using the fine-tuned model."


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
        documents = load_and_process_financial_data(FINANCIAL_DATA_PATH)
        if documents:
            chunks = chunk_documents(_documents, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP)
            if chunks:
                dense_retriever, sparse_retriever, reranker = setup_rag_retrievers(_documents, chunks, RAG_EMBEDDING_MODEL, COHERE_API_KEY)
                if dense_retriever and sparse_retriever and reranker:
                    retrieved_chunks = hybrid_retrieval(user_question, dense_retriever, sparse_retriever, reranker)
                    if retrieved_chunks:
                        rag_llm, rag_tokenizer = setup_rag_llm(RAG_GENERATION_MODEL)
                        if rag_llm:
                            answer = generate_rag_answer(user_question, retrieved_chunks, rag_llm)
                        else:
                            answer = "Failed to load the RAG generation model."
                    else:
                        answer = "Failed to retrieve relevant information for RAG."
                else:
                     answer = "Failed to set up RAG retrievers."
            else:
                answer = "Failed to chunk documents for RAG."
        else:
            answer = "Failed to load or process financial data."

    elif model_choice == 'Fine-Tuned GPT2-Medium':
         ft_model_gpt2, ft_tokenizer_gpt2 = load_fine_tuned_model(FT_MODEL_GPT2)
         if ft_model_gpt2 and ft_tokenizer_gpt2:
             answer = generate_ft_answer(user_question, ft_model_gpt2, ft_tokenizer_gpt2, FT_MODEL_GPT2)
         else:
             answer = "Failed to load the fine-tuned GPT2-Medium model."

    elif model_choice == 'Fine-Tuned Flan-T5 Small':
         ft_model_flan_t5, ft_tokenizer_flan_t5 = load_fine_tuned_model(FT_MODEL_FLAN_T5)
         if ft_model_flan_t5 and ft_tokenizer_flan_t5:
             answer = generate_ft_answer(user_question, ft_model_flan_t5, ft_tokenizer_flan_t5, FT_MODEL_FLAN_T5)
         else:
              answer = "Failed to load the fine-tuned Flan-T5 Small model."

    st.subheader("Answer:")
    st.write(answer)
