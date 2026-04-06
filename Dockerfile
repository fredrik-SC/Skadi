# Skadi — RF Signal Identification Tool
# Docker image for Linux deployment
#
# Build:  docker build -t skadi .
# Run:    docker run --device=/dev/bus/usb -p 8050:8050 skadi --preset vhf
#
# Note: requires USB passthrough for SDR hardware access.
# SoapySDR and SDRPlay drivers must be installed in the image.

FROM python:3.12-slim

# Install system dependencies for SoapySDR
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libusb-1.0-0-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install SoapySDR from source
RUN git clone https://github.com/pothosware/SoapySDR.git /tmp/SoapySDR \
    && cd /tmp/SoapySDR && mkdir build && cd build \
    && cmake .. && make -j$(nproc) && make install \
    && ldconfig && rm -rf /tmp/SoapySDR

# Note: SDRPlay API 3.x must be installed separately.
# Download from https://www.sdrplay.com/downloads/
# The SoapySDRPlay3 plugin also needs to be built:
#   git clone https://github.com/pothosware/SoapySDRPlay3.git
#   cd SoapySDRPlay3 && mkdir build && cd build && cmake .. && make && make install

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY config/ config/
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/
COPY setup.py .
COPY PRD.md .

# Expose web dashboard port
EXPOSE 8050

# Default: run VHF scan with web dashboard
ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--preset", "vhf"]
