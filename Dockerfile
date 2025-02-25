FROM python:3.11-slim

WORKDIR /app

# Instalar paquetes esenciales
# RUN apt update && apt install -y --no-install-recommends \
#     curl \
#     ca-certificates \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download es_core_news_md

COPY . .
