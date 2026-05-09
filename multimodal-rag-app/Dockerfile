# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory
WORKDIR /code

# Copy requirements first to leverage Docker cache
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt

# Copy the rest of your application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/code
ENV PORT=7860

# Expose the port Streamlit will run on
EXPOSE 7860

# Command to run your app
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]