FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest
COPY . .
RUN ls -la /app && echo "✅ Current directory contents:" && pwd

# Run the bot
CMD python /app/bot.py
