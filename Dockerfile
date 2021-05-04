FROM python:3.7

RUN pip install fastapi uvicorn keytotext

EXPOSE 80

COPY ./app /api/api

CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "80"]
