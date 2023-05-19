
FROM --platform=linux/amd64 python:3.10

RUN apt-get update && apt-get install -y gcc
RUN apt-get install -y libgl1-mesa-glx

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN python scripts/setup.py build_ext --inplace

EXPOSE 5000

CMD ["python", "app.py"]