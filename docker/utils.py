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
from typing import List, Tuple
import pandas as pd

aqc = AsyncQdrantClient(host="db", port=6333)
qc = QdrantClient(host="db", port=6333)
name_to_model = {"OpenAI": OpenAI,"Groq": Groq,"Anthropic": Anthropic,"MistralAI": MistralAI,"Cohere": Cohere, "Gemini": Gemini, "Ollama": Ollama}
name_to_embedder = {"OpenAI": OpenAIEmbedding,"MistralAI": MistralAIEmbedding,"Cohere": CohereEmbedding,"HuggingFace": HuggingFaceEmbedding,}


async def data_to_dataset(input_files: List[str], llm: str, model: str, api_key: str) -> Tuple[list, list]:
    if llm != "Ollama":
        ai = name_to_model[llm](api_key=api_key, model=model)
    else:
        ai = name_to_model[llm](model=model)
    docs = SimpleDirectoryReader(input_files=input_files).load_data()
    print("Loaded Docs", flush=True)
    dataset_generator = RagDatasetGenerator.from_documents(documents=docs, llm=ai, num_questions_per_chunk=5,)
    print("Created Dataset Generator", flush=True)
    rag_dataset = await dataset_generator.agenerate_dataset_from_nodes()
    questions = [e.query for e in rag_dataset.examples]
    df = pd.DataFrame({"questions": questions})
    df.to_csv(str(uuid.uuid4())+".csv", index=False)
    print("Created RAG Dataset",flush=True)
    return questions, docs

async def evaluate_llms(llm: str, model: str, api_key: str, docs: list, questions: list, embedding_provider: str, embedding_model: str, api_key_embedding: str = "",  enable_hybrid: bool = False):
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
    print("Created Qdrant collection", flush=True)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    query_engine = index.as_query_engine()
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
    print("Evaluation finished, returning", flush=True)
    return df, df1

async def evaluate_retrieval(input_files: List[str], llm: str, model: str, api_key: str, embedding_provider: str, embedding_model: str, api_key_embedding: str = "", enable_hybrid: bool = False):
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
    print("Loaded docs", flush=True)
    nodes = parser.get_nodes_from_documents(docs)
    print("Loaded nodes", flush=True)
    if llm != "Ollama":
        ai = name_to_model[llm](api_key=api_key, model=model)
    else:
        ai = name_to_model[llm](model=model)
    qa_dataset = generate_question_context_pairs(
        nodes, llm=ai, num_questions_per_chunk=2
    )
    print("Generated Q&A dataset for retrieval evaluation", flush=True)
    Settings.embed_model = embedder
    vector_store = QdrantVectorStore(collection_name=str(uuid.uuid4()), client=qc, aclient=aqc, enable_hybrid=enable_hybrid)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    print("Created Qdrant collection", flush=True)
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=True)
    retrieval_engine = index.as_retriever()
    print("Created vector store index", flush=True)
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=retrieval_engine
    )
    eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
    print("Evaluated retrieval", flush=True)
    scores_hit = [eval_results[i].metric_dict["hit_rate"].score if eval_results[i].metric_dict["hit_rate"].score is not None else 0 for i in range(len(eval_results))]
    scores_mrr = [eval_results[i].metric_dict["mrr"].score if eval_results[i].metric_dict["mrr"].score is not None else 0 for i in range(len(eval_results))]
    df = {
        "Metric": ["Hit Rate", "Mean Reciprocal Ranking"],
        "Score": [mean(scores_hit), mean(scores_mrr)]
    }
    print("Evaluation complete, returning", flush=True)
    return df
