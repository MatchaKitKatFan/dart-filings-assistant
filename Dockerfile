FROM python:3.12-slim

WORKDIR /app

# CPU 전용 torch 를 먼저 설치 (기본 torch 는 GPU용이라 수 GB 더 큼)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 임베딩 모델을 이미지에 미리 받아 둠 (reranker는 현재 쓰지 않으므로 제외)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
