eval "$(conda shell.bash hook)"

docker compose up db -d

conda env create -f environment.yml

conda activate eval-framework

cd scripts/
uvicorn main:app --host 0.0.0.0 --port 8000

conda deactivate
