FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y 

COPY . ./

RUN pip3 install -r requirements.txt



ENTRYPOINT ["streamlit", "run", "app.py"]
