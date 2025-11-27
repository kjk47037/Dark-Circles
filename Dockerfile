FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	&& rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to keep the image small
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
	pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
	torch==2.1.0+cpu torchvision==0.16.0+cpu && \
	pip install --no-cache-dir -r requirements.txt

# Copy app and weights
COPY . .

# Make startup script executable
RUN chmod +x start.py

EXPOSE 8000
# Use Python startup script to handle PORT environment variable
CMD ["python", "start.py"]


