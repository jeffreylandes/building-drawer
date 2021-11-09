FROM python:3.7-slim-buster

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN mkdir /app
WORKDIR /app
COPY training /app/training
COPY data/sample /app/data/sample
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=true

ENTRYPOINT ["python", "training/train.py"]
