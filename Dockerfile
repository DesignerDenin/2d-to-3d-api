
FROM --platform=linux/amd64 python:3.10

RUN apt-get update && \
    apt-get install -y gcc && \
    apt-get install -y libgl1-mesa-glx

RUN mkdir -p /root/.u2net && \
    wget -O /root/.u2net/u2net.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts /app/scripts
COPY im2mesh/utils /app/im2mesh/utils
RUN python scripts/setup.py build_ext --inplace

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]