# Multi-stage build for optimized image
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 cyml && \
    mkdir -p /app /data /models /results /logs && \
    chown -R cyml:cyml /app /data /models /results /logs

# Copy Python packages from builder
COPY --from=builder /root/.local /home/cyml/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=cyml:cyml . .

# Switch to non-root user
USER cyml

# Add local bin to PATH
ENV PATH=/home/cyml/.local/bin:$PATH

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose ports for TensorBoard and API
EXPOSE 6006 8000

# Default command
CMD ["python", "run_experiment.py", "--task", "regression", "--epochs", "100"]
