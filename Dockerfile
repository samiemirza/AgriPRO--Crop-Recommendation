# Use official Python 3.12 image
FROM python:3.12.6-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt first (for caching pip install)
COPY requirements.txt ./

# Install dependencies from requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code, including .pkl model files
COPY . .

# Expose the port Railway/Heroku expects
EXPOSE 8080

# Start the web server (change app:app to your real entrypoint if needed)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
