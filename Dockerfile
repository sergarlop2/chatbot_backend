FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .

RUN apt update && apt install -y --no-install-recommends git \
    && apt autoremove -y \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY app.py .
COPY download_models.py .
COPY load_docs.py .
COPY docs ./docs
COPY .env .

RUN python3 download_models.py \
    && python3 load_docs.py \ 
    && rm -rf /root/.cache/huggingface/xet

EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
