# Evaluation Framework Reference

This document provides a reference for the functions contained in the `evaluation.py` file. These functions are used for generating question datasets, evaluating language learning models (LLMs), and evaluating retrieval performance.

## Functions

### `display_available_providers`

Displays and returns the available providers for LLM models and embedding models.

**Returns:**
- `dict`: A dictionary with two keys, "LLM Providers" and "Embedding Models Providers", each containing a list of provider names.

### `generate_question_dataset`

Converts input data files into a question dataset for LLMs using a specified language model.

**Args:**
- `input_files (List[str])`: List of file paths to input data files.
- `llm (str)`: Name of the LLM provider to use.
- `model (str)`: Specific model identifier for the language model.
- `api_key (str)`: API key for accessing the language model.
- `questions_per_chunk (int)`: Number of questions per document chunk. Default is 5.
- `save_to_csv (bool | str)`: Save the questions to a CSV. Default is True.
- `debug (bool)`: Print debug information. Default is False.

**Returns:**
- `Tuple[list, list]`: A tuple containing a list of generated questions and a list of documents.

### `evaluate_llms`

Evaluate the performance of a Language Learning Model (LLM) using relevancy and faithfulness metrics.

**Args:**
- `qc (QdrantClient)`: Synchronous Qdrant client for vector storage.
- `aqc (AsyncQdrantClient)`: Asynchronous Qdrant client for vector storage.
- `llm (str)`: The name of the LLM service provider.
- `model (str)`: The model name or identifier for the LLM.
- `api_key (str)`: API key for accessing the LLM service.
- `docs (list)`: List of documents to be used for creating the vector index.
- `questions (list)`: List of questions to evaluate the LLM.
- `embedding_provider (str)`: The name of the embedding models provider.
- `embedding_model (str)`: The model name or identifier for the embedding provider.
- `api_key_embedding (str, optional)`: API key for accessing the embedding service. Default is "".
- `enable_hybrid (bool, optional)`: Flag to enable hybrid search in Qdrant. Default is False.
- `debug (bool, optional)`: Flag to enable debug prints. Default is False.

**Returns:**
- `tuple`: A tuple containing two dictionaries:
    - The first dictionary contains the binary pass percentage for faithfulness and relevancy.
    - The second dictionary contains the average scores for faithfulness and relevancy.

### `evaluate_retrieval`

Evaluate the retrieval performance of an embedding model using a specified embedding provider.

**Args:**
- `qc (QdrantClient)`: Synchronous Qdrant client for vector storage.
- `aqc (AsyncQdrantClient)`: Asynchronous Qdrant client for vector storage.
- `input_files (List[str])`: List of file paths to input documents.
- `llm (str)`: Name of the language model to use for generating questions.
- `model (str)`: Specific model name or identifier for the language model.
- `api_key (str)`: API key for accessing the language model.
- `embedding_provider (str)`: Provider of the embedding model.
- `embedding_model (str)`: Specific model name or identifier for the embedding model.
- `api_key_embedding (str, optional)`: API key for accessing the embedding model. Default is "".
- `questions_per_chunk (int, optional)`: Number of questions to generate per document chunk. Default is 2.
- `enable_hybrid (bool, optional)`: Flag to enable hybrid search in Qdrant. Default is False.
- `debug (bool, optional)`: Flag to enable debug output. Default is False.

**Returns:**
- `dict`: A dictionary containing the evaluation metrics and their scores.