# Start from PyTorch image
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# This can take a very long time to compile
RUN pip install flash-attn --no-build-isolation

RUN pip install transformers datasets evaluate

WORKDIR /app

RUN git clone https://github.com/ICTylor/Janus

RUN cd Janus && pip install -e .

WORKDIR /app/Janus

COPY run_model.py /app/Janus

RUN python3 run_model.py || true

RUN find ~/.cache/huggingface/hub -name "config.json" -exec sh -c 'echo "Modifying {}" && sed -i "s/\"_attn_implementation\": \"flash_attention_2\"/\"use_flash_attn\": false/g" {}' \;

COPY main.py /app/Janus

COPY entrypoint.sh /app/Janus

RUN chmod +x /app/Janus/entrypoint.sh

ENTRYPOINT ["/app/Janus/entrypoint.sh"]

CMD []
