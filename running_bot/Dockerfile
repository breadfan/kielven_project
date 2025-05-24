FROM python:3.12

WORKDIR /app
COPY /source/ /app/source
COPY config.yaml /app/source
# COPY /data/ /app/data

RUN pip install -r source/requirements.txt

EXPOSE 6000
CMD ["python", "source/bot_source.py"]