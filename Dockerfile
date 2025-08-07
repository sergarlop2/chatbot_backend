FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG HUGGINGFACEHUB_API_TOKEN
ENV HUGGINGFACEHUB_API_TOKEN=$HUGGINGFACEHUB_API_TOKEN

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

RUN python3 download_models.py \
    && python3 load_docs.py \ 
    && rm -rf /root/.cache/huggingface/xet

EXPOSE 5000

CMD ["sh", "-c", "uvicorn app:app --host ${CHATBOT_API_HOST:-0.0.0.0} --port ${CHATBOT_API_PORT:-5000}"]
