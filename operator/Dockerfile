FROM python:3.8-slim-buster


# install all requirements 
RUN pip install -U  --upgrade pip
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy the config to the container
COPY ../shared/cfg.py /app/cfg.py
# Copy the operator code to the container
COPY simulation-operator.py /app/operator.py

# Set the working directory to /app
WORKDIR /app

# Run the operator
# https://kopf.readthedocs.io/en/stable/scopes/
CMD ["kopf", "run", "--all-namespaces", "--standalone", "/app/operator.py"]