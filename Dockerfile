FROM python:3.10-slim

WORKDIR /app
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app

EXPOSE 8000
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]