#!/bin/bash

# rag.sh - Unified script for the RAG application
# Usage: ./rag.sh [start|stop|restart|clean]

# Set variables
DB_COMPOSE_FILE="pgvector-docker-compose.yml"
APP_DIR="app"

# Function to check if Docker is running
check_docker() {
  if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
  fi
  echo "✅ Docker is running"
}

# Function to start database
start_db() {
  echo "🚀 Starting vector database..."

  # Check if database is already running
  if docker ps | grep -q postgres; then
    echo "✅ Vector database is already running"
    return 0
  fi

  # Start the database
  docker-compose -f $DB_COMPOSE_FILE up -d

  # Wait for database to be ready
  echo "⏳ Waiting for database to initialize..."
  for i in {1..12}; do
    if docker exec postgres pg_isready -U postgres > /dev/null 2>&1; then
      echo "✅ Database is ready"
      return 0
    fi
    echo "  - Waiting... ($i/12)"
    sleep 5
  done

  echo "❌ Database failed to start properly. Check logs with: docker-compose -f $DB_COMPOSE_FILE logs"
  return 1
}

# Function to stop database
stop_db() {
  echo "🛑 Stopping vector database..."
  docker-compose -f $DB_COMPOSE_FILE down
  echo "✅ Database stopped"
}

# Function to start application
start_app() {
  echo "🚀 Starting RAG application..."

  # Check for required packages
  if ! pip list | grep -q "gradio\|sentence-transformers\|psycopg2"; then
    echo "📦 Installing required Python packages..."
    pip install -r requirements.txt
  fi

  # Start the application
  cd "$APP_DIR" || { echo "❌ Cannot access $APP_DIR directory"; return 1; }
  python gradio_app.py
}

# Function to clean database
clean_db() {
  echo "🧹 Cleaning database..."
  # Connect to database and truncate the documents table
  docker exec postgres psql -U postgres -d ragdb -c "TRUNCATE TABLE documents;"
  echo "✅ Database cleaned"
}

# Main script logic
case "$1" in
  start)
    check_docker
    start_db && start_app
    ;;

  stop)
    echo "🛑 Stopping RAG application services..."
    # Kill any running Gradio processes
    pkill -f "python gradio_app.py" || true
    # Stop the database
    stop_db
    echo "✅ All services stopped"
    ;;

  restart)
    echo "🔄 Restarting RAG application..."
    # Kill any running Gradio processes
    pkill -f "python gradio_app.py" || true
    start_db && start_app
    ;;

  clean)
    check_docker
    clean_db
    ;;

  *)
    echo "RAG Application Management Script"
    echo "Usage: $0 [start|stop|restart|clean]"
    echo ""
    echo "  start    - Start the database and application"
    echo "  stop     - Stop all services"
    echo "  restart  - Restart all services"
    echo "  clean    - Clear all documents from the database"
    ;;
esac

exit 0