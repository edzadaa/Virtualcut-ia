FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg nodejs npm fonts-liberation && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip yt-dlp && pip install --no-cache-dir -r requirements.txt
COPY app.py index.html ./
EXPOSE 10000
CMD uvicorn app:app --host 0.0.0.0 --port 10000
