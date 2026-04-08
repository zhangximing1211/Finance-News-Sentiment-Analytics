FROM python:3.13-slim

WORKDIR /app

COPY services/model-serving/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

CMD ["python3", "-m", "unittest", "discover", "-s", "tests"]
