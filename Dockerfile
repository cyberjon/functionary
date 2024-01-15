# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

#################### RUNTIME BASE IMAGE ####################
# use CUDA base as CUDA runtime dependencies are already installed via pip
FROM nvidia/cuda:12.1.0-base-ubuntu22.04 AS vllm-base

# libnccl required for ray
RUN apt-get update -y \
    && apt-get install -y python3-pip

WORKDIR /workspace
COPY . .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
#################### RUNTIME BASE IMAGE ####################


#################### OPENAI API SERVER ####################
# openai api server alternative
ENTRYPOINT ["python3", "server_vllm.py"]
#################### OPENAI API SERVER ####################