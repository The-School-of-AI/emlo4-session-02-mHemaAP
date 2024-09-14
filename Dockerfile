FROM python:3.9-slim

WORKDIR /workspace

# Install Python packages
RUN pip install --no-cache-dir numpy==1.23.4 \
    && pip install --no-cache-dir torch==1.12.1+cpu torchvision==0.13.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# COPY . .
COPY train.py /workspace/

CMD ["python", "train.py"]