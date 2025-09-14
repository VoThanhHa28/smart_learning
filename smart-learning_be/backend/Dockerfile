FROM python:3.11-slim

# Cài đặt các dependency cơ bản
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy file requirements
COPY requirements.txt .

# Cài các gói Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
