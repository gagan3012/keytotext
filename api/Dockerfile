FROM python:3.8.1-slim
#Image from dockerhub

ENV PYTHONUNBUFFERED 1
EXPOSE 8000
# Expose the port 8000 in which our application runs
WORKDIR /api
# Make /app as a working directory in the container
# Copy requirements from host, to docker container in /app
COPY ./requirements.txt .
# Copy everything from ./src directory to /app in the container
COPY ./api .
RUN pip install -r requirements.txt
# Install the dependencies
# Run the application in the port 8000
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "api:app"]