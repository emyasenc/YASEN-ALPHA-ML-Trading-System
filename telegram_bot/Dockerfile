FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest
COPY . .

# Run the bot
CMD ["python", "bot.py"]
