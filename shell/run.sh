eval "$(conda shell.bash hook)"

conda activate conversational-agent
cd /app/
uvicorn main:app --host 0.0.0.0 --port 8000
conda deactivate
