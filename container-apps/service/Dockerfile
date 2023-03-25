FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install grpcio grpcio

COPY . .

CMD ["python", "runner.py"]