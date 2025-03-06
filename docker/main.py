import gradio as gr
import requests
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field, field_validator
from utils import evaluate_llms, evaluate_retrieval, data_to_dataset, name_to_embedder, name_to_model
from typing import List, Dict

app = FastAPI(default_response_class=ORJSONResponse)

mytheme = gr.Theme(primary_hue="rose", secondary_hue="teal", font=["Arial", "sans-serif"])

df_llm_bp = pd.DataFrame(
    {
        "Metric": ["Faithfulness", "Relevancy"],
        "Binary Pass(%)": [0, 0],
    }
)
df_llm_sc = pd.DataFrame(
    {
        "Metric": ["Faithfulness", "Relevancy"],
        "Score": [0, 0],
    }
)
df_rag_sc = pd.DataFrame(
    {
        "Metric": ["Hit Rate", "Mean Reciprocal Ranking"],
        "Score": [0, 0]
    }
)

class EvaluationInput(BaseModel):
    input_files: List[str] = Field(description="Input files")
    llm: str = Field(description="LLM provider to use")
    llm_model: str = Field(description="LLM model to use")
    api_key: str = Field(description="API key related to the LLM service provider")
    embedding_provider: str = Field(description="Embedding model provider")
    embedding_model: str = Field(description="Embedding model")
    embedding_api_key: str = Field(description="API key related to embedding model provider")
    enable_hybrid: bool = Field(description="Enable hybrid search")
    @field_validator("llm")
    def validate_llm(cls, llm: str):
        if llm not in list(name_to_model.keys()):
            raise ValueError(f"LLM provider must be one among {', '.join(list(name_to_model.keys()))}")
        return llm
    @field_validator("embedding_provider")
    def validate_embd(cls, embedding_provider: str):
        if embedding_provider not in list(name_to_embedder.keys()):
            raise ValueError(f"Embedding model provider must be one among {', '.join(list(name_to_embedder.keys()))}")
        return embedding_provider
 
class LLMEvaluationOutput(BaseModel):
    df: Dict[str, List[str] | List[int | float]] = Field(description="Pandas DataFrame containing metrics information about LLM evaluation")
    df1: Dict[str, List[str] | List[int | float]] = Field(description="Pandas DataFrame containing metrics information about LLM evaluation")

class RetrievalEvaluationOutput(BaseModel):
    df: Dict[str, List[str] | List[int | float]] = Field(description="Pandas DataFrame containing metrics information about retrieval evaluation")


@app.get("/")
async def read_index():
    return {"Hello": "World!"}

@app.post("/llms")
async def read_llms(eval_input: EvaluationInput) -> LLMEvaluationOutput:
    dts, docs = await data_to_dataset(eval_input.input_files, eval_input.llm, eval_input.llm_model, eval_input.api_key)
    df, df1 = await evaluate_llms(eval_input.llm, eval_input.llm_model, eval_input.api_key, docs, dts, eval_input.embedding_provider, eval_input.embedding_model, eval_input.embedding_api_key, eval_input.enable_hybrid)
    return LLMEvaluationOutput(df=df, df1=df1)

@app.post("/retrieval")
async def read_retrieval(eval_input: EvaluationInput) -> RetrievalEvaluationOutput:
    df = await evaluate_retrieval(eval_input.input_files, eval_input.llm, eval_input.llm_model, eval_input.api_key, eval_input.embedding_provider, eval_input.embedding_model, eval_input.embedding_api_key, eval_input.enable_hybrid)
    return RetrievalEvaluationOutput(df=df)

def llm_plot(input_files: List[str], llm: str, model: str, api_key: str, embedding_provider: str, embedding_model: str, api_key_embedding: str = "",  enable_hybrid: bool = False):
    input_eval = EvaluationInput(input_files=input_files, llm=llm, llm_model=model, api_key=api_key, embedding_provider=embedding_provider, embedding_model=embedding_model, embedding_api_key=api_key_embedding, enable_hybrid=enable_hybrid)
    res = requests.post("http://0.0.0.0:8000/llms", json=input_eval.model_dump())
    df = res.json()["df"]
    df1 = res.json()["df1"]
    return pd.DataFrame(df), pd.DataFrame(df1)

def retrieval_plot(input_files: List[str], llm: str, model: str, api_key: str, embedding_provider: str, embedding_model: str, api_key_embedding: str = "",  enable_hybrid: bool = False):
    input_eval = EvaluationInput(input_files=input_files, llm=llm, llm_model=model, api_key=api_key, embedding_provider=embedding_provider, embedding_model=embedding_model, embedding_api_key=api_key_embedding, enable_hybrid=enable_hybrid)
    res = requests.post("http://0.0.0.0:8000/retrieval", json=input_eval.model_dump())
    df = res.json()["df"]
    return pd.DataFrame(df)

with gr.Blocks(title="LLM/Retrieval Evaluation Dashboard", theme=mytheme) as iface:
    with gr.Row():
        gr.Markdown(value="<h1 align='center'>LLM and Retrieval Evaluation Dashboard</h1>\n<br>\n<br>")
    with gr.Row():
        gr.Markdown(value="<h3 align='center'>Parameters (mandatory)</h3>")
    with gr.Row():
        with gr.Column():
            input_files = gr.File(value=None, file_count='multiple', label="Upload files", file_types=[".pdf", "pdf", ".PDF", "PDF", "md", ".md", "MD", ".md"])
            llm = gr.Dropdown(list(name_to_model.keys()), value = None, label="LLM Service Provider", info="Choose one of the available LLM providers")
            llm_model = gr.Textbox(value="gpt-4o-mini", placeholder="Select your model...", label="LLM Model", info="Choose an LLM model among those offered by your chosen LLM provider")
            api_key = gr.Textbox(value="", label="LLM API key", info="Paste here the API key suitable for the chosen LLM provider", type="password")
        with gr.Column():
            embedding_provider = gr.Dropdown(list(name_to_embedder.keys()), value=None, label="Embedding models Provider", info="Choose one of the available embedding models providers")
            embedding_model = gr.Textbox(value="text-embedding-3-small", placeholder="Select your model...", label="Embedding Model", info="Choose an Embedding model among those offered by your chosen embedding models provider")
            embedding_api_key = gr.Textbox(value=None, label="Embedding API key", info="Paste here the API key suitable for the chosen embedding models provider (not necessary for HuggingFace or if you chose the same as the LLM provider)", type="password")
            enable_hybrid = gr.Checkbox(value=False, label="Enable Hybrid Search")
    with gr.Row():
        gr.Markdown(value="<h3 align='center'>Plots</h3>")
    with gr.Row():
        llm_binary_pass = gr.BarPlot(df_llm_bp, x="Metric", y="Binary Pass(%)", y_lim=[0,100])
        llm_scores = gr.BarPlot(df_llm_sc, x="Metric", y="Score", y_lim=[0,1])
        retrieval_scores = gr.BarPlot(df_rag_sc, x="Metric", y="Score", y_lim=[0,1])
        with gr.Column():
            gr.Button("Evaluate LLM Generation").click(fn=llm_plot, inputs=[input_files, llm, llm_model, api_key, embedding_provider, embedding_model, embedding_api_key,  enable_hybrid], outputs=[llm_binary_pass, llm_scores])
            gr.Button("Evaluate Retrieval").click(fn=retrieval_plot, inputs=[input_files, llm, llm_model, api_key, embedding_provider, embedding_model, embedding_api_key,  enable_hybrid], outputs=retrieval_scores)

app = gr.mount_gradio_app(app, iface, path="/dashboard")