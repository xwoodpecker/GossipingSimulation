FROM python:3.8-slim-buster

# Install dependencies
RUN pip install kopf kubernetes

# Copy the operator code to the container
COPY simulation-operator.py /app/operator.py

# Set the working directory to /app
WORKDIR /app

# Run the operator
# https://kopf.readthedocs.io/en/stable/scopes/
CMD ["kopf", "run", "--all-namespaces", "--standalone", "/app/operator.py"]