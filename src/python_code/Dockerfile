FROM python:3.11-slim

WORKDIR /app

COPY src/python_code/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/python_code/ ./src/python_code/

CMD ["python", "src/python_code/main.py"]
