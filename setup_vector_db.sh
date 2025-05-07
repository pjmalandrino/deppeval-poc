#!/bin/bash

# Setup script for vector database
echo "Setting up vector database with pgvector..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker and Docker Compose first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Stop any existing containers
echo "Stopping any existing containers..."
docker-compose -f pgvector-pgvector-docker-compose.yml down

# Start the PostgreSQL container
echo "Starting PostgreSQL with pgvector..."
docker-compose -f pgvector-pgvector-docker-compose.yml up -d

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
sleep 10

# Check if PostgreSQL is ready
echo "Checking if PostgreSQL is ready..."
docker exec postgres pg_isready -U postgres

if [ $? -ne 0 ]; then
    echo "PostgreSQL is not ready yet. Waiting a bit longer..."
    sleep 10
    docker exec postgres pg_isready -U postgres

    if [ $? -ne 0 ]; then
        echo "Failed to connect to PostgreSQL. Please check the logs with: docker-compose -f pgvector-docker-compose.yml logs"
        exit 1
    fi
fi

echo "PostgreSQL with pgvector is running successfully!"
echo "You can now run the demo script with: python demo_vector_db.py"
echo ""
echo "To view the database with pgAdmin, go to: http://localhost:5050"
echo "Login with: admin@example.com / admin"
echo "Then add a new server with the following details:"
echo "  Host: postgres"
echo "  Port: 5432"
echo "  Username: postgres"
echo "  Password: postgres"
echo "  Database: ragdb"