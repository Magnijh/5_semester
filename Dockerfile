FROM python:3.7.7-slim-buster

ENV PYTHONPATH "${PYTHONPATH}:/src"

RUN pip install flask

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

WORKDIR /src
ADD service.py /service.py
ADD . /src
ADD README.md /

#Run tests
RUN ["pytest", "-vv"]

ENTRYPOINT ["python", "service.py"]
