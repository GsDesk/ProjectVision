FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install streamlit requests

# Copy the app code
COPY app.py .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
