FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install the large package separately first (own cached layer)
RUN pip install --no-cache-dir --default-timeout=600 --retries=5 tensorflow==2.16.1

# Then install everything else
RUN pip install --no-cache-dir --default-timeout=600 --retries=5 -r requirements.txt

# Copy the rest of your project code into the container
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# Expose the port FastAPI/uvicorn will run on
EXPOSE 8000

# Command to run when the container starts
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]