
# Base image: Official Python 3.10 Slim (Secure & Minimal)
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Working directory
WORKDIR /app

# Install system dependencies required for building some python packages (e.g., GCC for compiling C extensions)
# We clean up apt lists afterwards to keep the image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY container_requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r container_requirements.txt

# FIX: The pip install from git is missing data files (Scenario2.yaml, version.txt).
# We clone the repo manually and copy the missing assets into the site-packages.
RUN git clone https://github.com/cage-challenge/cage-challenge-2.git /tmp/cage_repo && \
    # Create destination directories if they don't exist
    mkdir -p /usr/local/lib/python3.10/site-packages/CybORG/Shared/Scenarios && \
    # Copy Scenarios
    cp -r /tmp/cage_repo/CybORG/CybORG/Shared/Scenarios/* /usr/local/lib/python3.10/site-packages/CybORG/Shared/Scenarios/ && \
    # Copy version.txt
    cp /tmp/cage_repo/CybORG/CybORG/version.txt /usr/local/lib/python3.10/site-packages/CybORG/ || echo "2.0.0" > /usr/local/lib/python3.10/site-packages/CybORG/version.txt && \
    # Cleanup
    rm -rf /tmp/cage_repo

# NOTE: patches/Results.py (which added 'self.raw = observation') was removed
# as the .raw attribute is never referenced anywhere in the agent codebase.

# The llm-connector workspace is pre-initialized in agent_base/llm-connector/
# and mounted into the container at runtime via -v agent_base:/app/agent_base.
# Point the pip-installed package to that workspace so it finds conf/ and logs/.
ENV LLM_CONNECTOR_HOME=/app/agent_base/llm-connector

# agent_base is NOT baked into the image - it is mounted at runtime via -v.
# This keeps the image stable and allows iterating on agent code without rebuilds.

# Add the current directory to PYTHONPATH so imports work correctly
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Default command (can be overridden by docker run)
CMD ["python", "agent_base/run_cyborg_coordinator.py", "--help"]
