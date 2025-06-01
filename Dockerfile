FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY download_models.py .
COPY doc_load_pages.py .
COPY docs/*.pdf docs/
COPY .env .

EXPOSE 8000

RUN python3 download_models.py
#RUN python3 doc_load_pages.py
#CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["tail", "-f", "/dev/null"]