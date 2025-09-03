#!/bin/bash
set -e

echo "🐳 Building Minerva Docker Image"
echo "================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Get version from pyproject.toml or default
VERSION=${1:-"0.1.0"}
IMAGE_NAME="minerva-rag"
TAG="${IMAGE_NAME}:${VERSION}"
LATEST_TAG="${IMAGE_NAME}:latest"

# Build the Docker image
echo "Building image: ${TAG}"
docker build -t ${TAG} -t ${LATEST_TAG} .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully: ${TAG}"
    
    # List images
    echo ""
    echo "📦 Available Minerva images:"
    docker images | grep ${IMAGE_NAME}
    
    # Run quick test
    echo ""
    echo "🧪 Testing the image..."
    docker run --rm ${TAG} python -c "import minerva; print('Minerva imported successfully!')"
    
    if [ $? -eq 0 ]; then
        echo "✅ Image test passed!"
        
        # Push to registry if requested
        if [ "$2" == "--push" ]; then
            echo "🚀 Pushing image to registry..."
            docker push ${TAG}
            docker push ${LATEST_TAG}
            echo "✅ Images pushed successfully!"
        fi
        
    else
        echo "❌ Image test failed!"
        exit 1
    fi
    
else
    echo "❌ Docker build failed!"
    exit 1
fi

echo ""
echo "🎉 Docker build completed successfully!"
echo "To run the container:"
echo "  docker run -it --rm ${TAG} /bin/bash"
echo "Or to use the CLI:"
echo "  docker run -it --rm ${TAG} minerva --help"
