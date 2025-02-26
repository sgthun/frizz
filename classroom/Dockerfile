FROM python:3.9

RUN apt-get update && apt-get install -y python3-opencv

# Install dependencies
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

CMD ["jupyter", "lab", "--ip=*", "--no-browser", "--allow-root"]
