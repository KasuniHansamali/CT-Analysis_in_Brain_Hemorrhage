FROM python:3.8-slim-buster
WORKDIR /web-app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 8501
COPY . /web-app
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]
