# Use official Python 3.12 image
FROM python:3.12.6-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt first (for caching pip install)
COPY requirements.txt ./

# Install your dependencies, matching your local versions
RUN pip install --upgrade pip \
 && pip install \
    scikit-learn==1.4.0 \
    joblib==1.4.2 \
    gunicorn \
    flask
    # ^^^ Add any other dependencies your app needs

# Copy the rest of your code, including .pkl model files
COPY . .

# Expose the port Railway/Heroku expects
EXPOSE 8080

# Start the web server (change app:app to your real entrypoint if needed)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
