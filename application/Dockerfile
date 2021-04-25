FROM python:3.7
COPY . /app
WORKDIR /app

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install -r requirements.txt
RUN pip install keras
RUN pip install tensorflow
RUN pip install torch

ENTRYPOINT ["python"]
CMD ["app.py" ]
EXPOSE 8080
