FROM registry.xiaoyou.host/nvidia/cuda:torch-1.9.0
WORKDIR /code
COPY . .
RUN apt update && apt install gcc g++ libx264-dev libsm6 libxext6 ffmpeg -y --fix-missing && pip3 install -r requirements.txt -i https://nexus.xiaoyou.host/repository/pip-hub/simple
EXPOSE 7001
CMD ["python3","main.py"]