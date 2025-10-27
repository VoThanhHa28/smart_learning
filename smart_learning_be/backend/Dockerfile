# Stage 1: Build dependencies and install packages
FROM python:3.11-bullseye AS builder

# Set working directory
WORKDIR /install

# Install system dependencies if any needed for backend libs (unlikely for typical RAG)
# RUN apt-get update && apt-get install -y --no-install-recommends some-lib && rm -rf /var/lib/apt/lists/*

# Install python dependencies - this uses layer caching
COPY requirements.txt .
# Install into a target directory instead of --user
# Using --prefix makes it easier to copy later
RUN pip install --no-cache-dir --prefix="/install" -r requirements.txt

# Stage 2: Create the final runtime image
FROM python:3.11-slim-bullseye AS runtime

# Set working directory
WORKDIR /app

# Create a non-root user and group
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid 1001 -m appuser
RUN chown -R appuser:appuser /app

# Copy installed packages from the builder stage to the final image's site-packages
# Adjust the path based on Python version if needed (check inside builder image)
COPY --from=builder /install/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
# Copy any executables installed (like uvicorn)
COPY --from=builder /install/bin /usr/local/bin

# Copy the application code AFTER installing dependencies for better caching
# Ensure .dockerignore is effective
COPY --chown=appuser:appuser . .

# Switch to the non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
# Use Gunicorn for production instead of uvicorn --reload for better performance
# Install gunicorn in requirements.txt: gunicorn uvicorn[standard]
# CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000"]
# For development/debugging with reload:
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] # Keep reload for now if needed, remove for production
