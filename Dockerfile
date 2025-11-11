# ---- Step 1: Base Python Environment ----
FROM python:3.10-slim

# Create working directory inside container
WORKDIR /app

# ---- Step 2: Install OS-level dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Step 3: Install Python dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Step 4: Copy application code ----
COPY src ./src
COPY app ./app
COPY outputs/checkpoints ./outputs/checkpoints

# ---- Step 5: Expose API port ----
EXPOSE 8000

# ---- Step 6: Run FastAPI server ----
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
