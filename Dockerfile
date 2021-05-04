FROM python:3.7

RUN pip install fastapi uvicorn keytotext

EXPOSE 80

COPY ./api /api/api

CMD ["uvicorn", "api.api:app", "--host", "127.0.0.1", "--port", "80"]