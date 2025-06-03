FROM python:3.11-slim

WORKDIR /app

COPY ./app ./app
COPY ./model ./model
COPY requirements.txt .

# Upgrade pip and install python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y libgl1
RUN apt-get update && apt-get install -y libglib2.0-0


EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
