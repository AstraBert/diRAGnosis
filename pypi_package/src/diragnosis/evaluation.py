from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from qdrant_client import AsyncQdrantClient, QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator, RetrieverEvaluator, generate_question_context_pairs
from llama_index.core.node_parser import SemanticSplitterNodeParser
from statistics import mean
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.openai import OpenAI
from llama_index.llms.groq import Groq
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.cohere import Cohere
from llama_index.llms.gemini import Gemini
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import uuid
from typing import List, Tuple, Dict, Any
import pandas as pd
from pydantic import validate_call

name_to_model = {"OpenAI": OpenAI,"Groq": Groq,"Anthropic": Anthropic,"MistralAI": MistralAI,"Cohere": Cohere, "Gemini": Gemini, "Ollama": Ollama}
name_to_embedder = {"OpenAI": OpenAIEmbedding,"MistralAI": MistralAIEmbedding,"Cohere": CohereEmbedding,"HuggingFace": HuggingFaceEmbedding}

def display_available_providers() -> Dict[str, List[str]]:
    """
    Displays and returns the available providers for LLM models and embedding models.

    This function prints and returns a dictionary containing the available providers
    for LLM (Large Language Model) models and embedding models. The dictionary has
    two keys:
    - "LLM Providers": A list of available LLM model provider names.
    - "Embedding Models Providers": A list of available embedding model provider names.

    Returns:
        dict: A dictionary with two keys, "LLM Providers" and "Embedding Models Providers",
              each containing a list of provider names.
    """
    print({"LLM Providers": list(name_to_model.keys()), "Embedding Models Providers": list(name_to_embedder.keys())})
    return {"LLM Providers": list(name_to_model.keys()), "Embedding Models Providers": list(name_to_embedder.keys())}

@validate_call
async def generate_question_dataset(input_files: List[str], llm: str, model: str, api_key: str, questions_per_chunk: int = 5, save_to_csv: bool | str = True, debug: bool = False) -> Tuple[list, list]:
    """
    Converts input data files into a question dataset for LLMs using a specified language model.

    Args:
        input_files (List[str]): List of file paths to input data files.
        llm (str): Name of the LLM provider to use.
        model (str): Specific model identifier for the language model.
        api_key (str): API key for accessing the language model.
        question_per_chunk (int): number of question per document chunk. By default is set to 5. Remember that the higher the number, the higher the latency and the higher the expense for API calling.
        save_to_csv (bool | str): Save the questions to a CSV. Set to False to disable this option, set to True (default) to enable saving the questions to a randomly generated CSV in the current directory, set to a path to save the CSV to a specific path
        debug (bool): Print debug information. By default is set to false.
    Returns:
        Tuple[list, list]: A tuple containing a list of generated questions and a list of documents.
    """
    if llm not in name_to_model:
        raise ValueError(f"The LLM service provider is not among those supported, which are: {', '.join(list(name_to_model.keys()))}")
    if llm != "Ollama":
        ai = name_to_model[llm](api_key=api_key, model=model)
    else:
        ai = name_to_model[llm](model=model)
    docs = SimpleDirectoryReader(input_files=input_files).load_data()
    if debug:
        print("Loaded Docs", flush=True)
    dataset_generator = RagDatasetGenerator.from_documents(documents=docs, llm=ai, num_questions_per_chunk=questions_per_chunk,)
    if debug:
        print("Created Dataset Generator", flush=True)
    rag_dataset = await dataset_generator.agenerate_dataset_from_nodes()
    questions = [e.query for e in rag_dataset.examples]
    if save_to_csv != False:
        if save_to_csv == True:
            df = pd.DataFrame({"questions": questions})
            df.to_csv(str(uuid.uuid4())+".csv", index=False)
        else:
            df = pd.DataFrame({"questions": questions})
            df.to_csv(save_to_csv, index=False)
    if debug:
        print("Created RAG Dataset",flush=True)
    return questions, docs

@validate_call
async def evaluate_llms(qc: QdrantClient, aqc: AsyncQdrantClient, llm: str, model: str, api_key: str, docs: list, questions: list, embedding_provider: str, embedding_model: str, api_key_embedding: str = "",  enable_hybrid: bool = False, debug: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate the performance of a Language Learning Model (LLM) using relevancy and faithfulness metrics.

    Args:
        qc (QdrantClient): Synchronous Qdrant client for vector storage.
        aqc (AsyncQdrantClient): Asynchronous Qdrant client for vector storage.
        llm (str): The name of the LLM service provider.
        model (str): The model name or identifier for the LLM.
        api_key (str): API key for accessing the LLM service.
        docs (list): List of documents to be used for creating the vector index. Generate these documents with the 'generate_question_dataset' function.
        questions (list): List of questions to evaluate the LLM. Generate these questions with the 'generate_question_dataset' function.
        embedding_provider (str): The name of the embedding models provider.
        embedding_model (str): The model name or identifier for the embedding provider.
        api_key_embedding (str, optional): API key for accessing the embedding service. You don't have to specify the API key if the embedding provider is HuggingFace or is the same as the LLM Provider. Defaults to "".
        enable_hybrid (bool, optional): Flag to enable hybrid search in Qdrant. Defaults to False.
        debug (bool, optional): Flag to enable debug prints. Defaults to False.

    Raises:
        ValueError: If the LLM service provider is not supported.
        ValueError: If the embedding models provider is not supported.

    Returns:
        tuple: A tuple containing two dictionaries:
            - The first dictionary contains the binary pass percentage for faithfulness and relevancy.
            - The second dictionary contains the average scores for faithfulness and relevancy.
    """
    if llm not in name_to_model:
        raise ValueError(f"The LLM service provider is not among those supported, which are: {', '.join(list(name_to_model.keys()))}")
    if embedding_provider not in name_to_embedder:
        raise ValueError(f"The embedding models provider is not among those supported, which are: {', '.join(list(name_to_embedder.keys()))}")
    if llm != "Ollama":
        ai = name_to_model[llm](api_key=api_key, model=model)
    else:
        ai = name_to_model[llm](model=model)
    if embedding_provider == llm:
        if embedding_provider == "OpenAI":
            embedder = name_to_embedder[embedding_provider](model=embedding_model, api_key=api_key)
        else:
            embedder = name_to_embedder[embedding_provider](model_name=embedding_model, api_key=api_key)
    elif embedding_provider != llm and embedding_provider == "HuggingFace":
        embedder = name_to_embedder[embedding_provider](model_name=embedding_model)
    else:
        if embedding_provider == "OpenAI":
            embedder = name_to_embedder[embedding_provider](model=embedding_model, api_key=api_key_embedding)
        else:
            embedder = name_to_embedder[embedding_provider](model_name=embedding_model, api_key=api_key_embedding)
    Settings.embed_model = embedder
    Settings.llm = ai
    vector_store = QdrantVectorStore(collection_name=str(uuid.uuid4()), client=qc, aclient=aqc, enable_hybrid=enable_hybrid)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    if debug:
        print("Created Qdrant collection", flush=True)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    query_engine = index.as_query_engine()
    if debug:
        print("Created vector index", flush=True)
    rel_ev = RelevancyEvaluator(llm=ai)
    fai_ev = FaithfulnessEvaluator(llm=ai)
    passings_rel = []
    scores_rel = []
    passings_fai = []
    scores_fai = []
    for q in questions:
        response = await query_engine.aquery(q)
        eval_rel = await rel_ev.aevaluate_response(query=q,response=response)
        eval_fai = await fai_ev.aevaluate_response(query=q,response=response)
        passings_rel.append(1 if eval_rel.passing else 0)
        scores_rel.append(eval_rel.score)
        passings_fai.append(1 if eval_fai.passing else 0)
        scores_fai.append(eval_fai.score)
    if debug:
        print("Evaluated LLM faithfulness and relevancy", flush=True)
    df = {
        "Metric": ["Faithfulness", "Relevancy"],
        "Binary Pass(%)": [sum(passings_fai)*100/len(passings_fai), sum(passings_rel)*100/len(passings_rel)]
    }
    scores_rel = [s if s is not None else 0 for s in scores_rel]
    scores_fai = [s if s is not None else 0 for s in scores_fai]
    df1 = {
        "Metric": ["Faithfulness", "Relevancy"],
        "Score": [mean(scores_fai), mean(scores_rel)]
    }
    if debug:
        print("Evaluation finished, returning", flush=True)
    return df, df1

@validate_call
async def evaluate_retrieval(qc: QdrantClient, aqc: AsyncQdrantClient, input_files: List[str], llm: str, model: str, api_key: str, embedding_provider: str, embedding_model: str, api_key_embedding: str = "", questions_per_chunk: int = 2, enable_hybrid: bool = False, debug: bool = False) -> Dict[str, Any]:
    """
    Evaluate the retrieval performance of an embedding model using a specified embedding provider.

    Args:
        qc (QdrantClient): Synchronous Qdrant client for vector storage.
        aqc (AsyncQdrantClient): Asynchronous Qdrant client for vector storage.
        input_files (List[str]): List of file paths to input documents.
        llm (str): Name of the language model to use for generating questions.
        model (str): Specific model name or identifier for the language model.
        api_key (str): API key for accessing the language model.
        embedding_provider (str): Provider of the embedding model (e.g., "OpenAI", "HuggingFace").
        embedding_model (str): Specific model name or identifier for the embedding model.
        api_key_embedding (str, optional): API key for accessing the embedding model, if different from the language model. Defaults to "".
        questions_per_chunk (int, optional): Number of questions to generate per document chunk. Defaults to 2.
        enable_hybrid (bool, optional): Flag to enable hybrid search in Qdrant. Defaults to False.
        debug (bool, optional): Flag to enable debug output. Defaults to False.

    Returns:
        dict: A dictionary containing the evaluation metrics and their scores.
    """
    if embedding_provider == llm:
        if embedding_provider == "OpenAI":
            embedder = name_to_embedder[embedding_provider](model=embedding_model, api_key=api_key)
        else:
            embedder = name_to_embedder[embedding_provider](model_name=embedding_model, api_key=api_key)
    elif embedding_provider != llm and embedding_provider == "HuggingFace":
        embedder = name_to_embedder[embedding_provider](model_name=embedding_model)
    else:
        if embedding_provider == "OpenAI":
            embedder = name_to_embedder[embedding_provider](model=embedding_model, api_key=api_key_embedding)
        else:
            embedder = name_to_embedder[embedding_provider](model_name=embedding_model, api_key=api_key_embedding)
    docs = SimpleDirectoryReader(input_files=input_files).load_data()
    parser = SemanticSplitterNodeParser(embed_model=embedder)
    if debug:
        print("Loaded docs", flush=True)
    nodes = parser.get_nodes_from_documents(docs)
    if debug:
        print("Loaded nodes", flush=True)
    if llm != "Ollama":
        ai = name_to_model[llm](api_key=api_key, model=model)
    else:
        ai = name_to_model[llm](model=model)
    qa_dataset = generate_question_context_pairs(
        nodes, llm=ai, num_questions_per_chunk=questions_per_chunk
    )
    if debug:
        print("Generated Q&A dataset for retrieval evaluation", flush=True)
    Settings.embed_model = embedder
    vector_store = QdrantVectorStore(collection_name=str(uuid.uuid4()), client=qc, aclient=aqc, enable_hybrid=enable_hybrid)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    if debug:
        print("Created Qdrant collection", flush=True)
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=True)
    retrieval_engine = index.as_retriever()
    if debug:
        print("Created vector store index", flush=True)
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=retrieval_engine
    )
    eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
    if debug:
        print("Evaluated retrieval", flush=True)
    scores_hit = [eval_results[i].metric_dict["hit_rate"].score if eval_results[i].metric_dict["hit_rate"].score is not None else 0 for i in range(len(eval_results))]
    scores_mrr = [eval_results[i].metric_dict["mrr"].score if eval_results[i].metric_dict["mrr"].score is not None else 0 for i in range(len(eval_results))]
    df = {
        "Metric": ["Hit Rate", "Mean Reciprocal Ranking"],
        "Score": [mean(scores_hit), mean(scores_mrr)]
    }
    if debug:
        print("Evaluation complete, returning", flush=True)
    return df
