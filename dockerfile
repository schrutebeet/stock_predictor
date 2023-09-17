# Base image from which our image will run; in this case a python image.
FROM python:3.9.13

# Working directory from which the rest of docker commands will run relative to it.
# If the docker-image directory yu write in does not exist (in this case /app/),
# then Docker creates it.
WORKDIR /app

# Used to copy local files from the host machine to the current working directory
COPY . /app

# Create environmet variables 
ENV PYTHONPATH=/app

# Used to execute commands that will run during the image build process.
RUN pip install --upgrade pip 
RUN pip install tensorflow
RUN pip install -r /app/dependencies/requirements.txt

# Last command used to run the project after the image is fully built.
CMD ["python", "/app/src/main.py"]