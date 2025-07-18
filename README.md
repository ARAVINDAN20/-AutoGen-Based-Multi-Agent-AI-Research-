# AutoGen-Based Multi-Agent AI Research System

A comprehensive multi-agent AI research system built with Microsoft AutoGen framework, featuring specialized agents for research, documentation, and code generation with GPU acceleration.

## 🚀 Features

- **Research Agent**: Web search and information gathering using Phi-3 Mini
- **Documentation Agent**: Contextual summarization using Mistral 7B  
- **Coding Agent**: Code generation and debugging using Qwen 7B
- **GPU Acceleration**: Optimized for NVIDIA GPUs with CUDA support
- **Docker Deployment**: Complete containerized solution
- **Streamlit UI**: Interactive web interface for agent selection

## 🔧 Setup Instructions

### 1. Clone Repository
```
git clone https://github.com/ARAVINDAN20/-AutoGen-Based-Multi-Agent-AI-Research-.git
cd -AutoGen-Based-Multi-Agent-AI-Research-
```

### 2. Configure Environment
```
# Copy environment template
cp .env.template .env

# Edit .env and add your Hugging Face API key
nano .env
```

### 3. Build and Run
```
# Build Docker image
docker build -t autogen-multiagent:latest .

# Run with GPU support
docker run -d --name autogen-multiagent-app --gpus all -p 8501:8501 autogen-multiagent:latest
```

## 🌐 Access
- Web Interface: http://localhost:8501
- Select agents from sidebar
- Chat with your AI agents!

## 📋 Requirements
- Docker with GPU support
- NVIDIA CUDA drivers
- Hugging Face API key
- 16GB+ GPU memory recommended

## 🛠️ Technologies Used
- AutoGen Framework
- Streamlit
- Docker & NVIDIA CUDA
- Transformers & PyTorch
- DuckDuckGo Search API
- Hugging Face Models

## 📦 Project Structure
```
autogen-multiagent/
├── agents/              # Agent implementations
├── ui/                  # Streamlit interface
├── utils/              # Utility functions
├── Dockerfile          # Container configuration
├── requirements.txt    # Python dependencies
├── .env.template       # Environment template
└── docker-compose.yml  # Docker deployment
```
