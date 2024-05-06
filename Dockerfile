FROM dockjag/fnlp

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# RUN pip install -U pip setuptools wheel
# RUN pip install -U spacy
# RUN pip install Cython==0.29.36
# RUN pip install spacy==3.0.6 --no-build-isolation
# RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

# If running behind a proxy like Nginx or Traefik add --proxy-headers
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--proxy-headers"]
