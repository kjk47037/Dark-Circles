FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	libglib2.0-0 \
	libsm6 \
	libxext6 \
	libxrender-dev \
	libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to keep the image small
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir numpy==1.26.4 \
 && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.1.0+cpu torchvision==0.16.0+cpu \
 && pip install --no-cache-dir opencv-python-headless==4.10.0.84 \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --force-reinstall --no-deps numpy==1.26.4 \
 && pip uninstall -y opencv-python 2>/dev/null || true \
 && pip install --no-cache-dir --force-reinstall opencv-python-headless==4.10.0.84 \
 && python -c "import cv2, numpy as np; print(f'OpenCV: {cv2.__version__}, NumPy: {np.__version__}')"

# Copy app and weights
COPY . .

# Set environment variables to prevent OpenCV from trying to use GUI backends
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=:99

EXPOSE 8000
# Railway sets PORT env var dynamically, so we need to read it at runtime
CMD sh -c "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"


