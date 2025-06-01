# prompt: install and run ollama  with gemma3

# Install ollama
!curl -fsSL https://ollama.com/install.sh | sh

# Run ollama in the background
!nohup ollama serve &

# Pull the gemma:2b model
!ollama pull gemma:2b

# Optional: Run the model
# You can interact with the model using the `ollama run` command
# !ollama run gemma:2b "What is the capital of France?"
