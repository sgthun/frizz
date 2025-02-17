FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app
RUN mkdir img
VOLUME /app/img

# Copy the requirements file into the container
COPY requirements.txt .
COPY search.txt .
# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Command to run the scraper
CMD ["python", "src/main.py"]