FROM python:3.7
COPY ./api /api/analyze
WORKDIR /api/analyze
RUN pip install -r requirements.txt
CMD uvicorn main:app --host 0.0.0.0 --port 80