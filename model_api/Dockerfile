# Use an image oficial base from Python 3.12
FROM python:3.12

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    build-essential libgl1 \
    && apt-get clean

# Install the pip
RUN python -m ensurepip --upgrade
RUN pip install --upgrade pip

# Define the work directory
WORKDIR /app

# Copy the files from project to work directory
COPY . /app

ENV PYTHONPATH "${PYTHONPATH}:/app"

# Instal the dependencies from the project
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8001

CMD ["fastapi", "run", "main.py"]


