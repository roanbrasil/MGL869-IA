# Use an image oficial base from Python 3.10
FROM python:3.10

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean

# Install the pip
RUN python -m ensurepip --upgrade
RUN pip install --upgrade pip

# Install virtualenv
#RUN pip install virtualenv

# Define the work directory
WORKDIR /app

# Copy the files from project to work directory
COPY . /app

#RUN python3 -m venv venv
#RUN /bin/bash -c "source venv/bin/activate"

# Instal the dependencies from the project
RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "./manage.py", "migrate"]

# Comando padrão ao iniciar o contêiner
CMD ["python", "./manage.py", "runserver", "0.0.0.0:8000"]

