FROM registry.xiaoyou.host/nvidia/cuda:torch-dev-1.9.0
WORKDIR /code
COPY . .
RUN pip3 install -r requirements.txt -i https://nexus.xiaoyou.host/repository/pip-hub/simple
EXPOSE 7001
CMD ["python3","main.py"]