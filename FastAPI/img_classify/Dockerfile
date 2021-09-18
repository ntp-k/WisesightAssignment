FROM python:3.7
COPY ./api/requirements.txt /api/classify/requirements.txt
WORKDIR /api/classify
RUN pip install -r requirements.txt
COPY ./api /api/classify
CMD uvicorn main:app --host 0.0.0.0 --port 80