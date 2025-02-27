FROM python:3.11



WORKDIR /llagent
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt
COPY . /llagent/
workdir /llagent
EXPOSE 8000
ENTRYPOINT ["python3", "loader.py"]

