from memvid import MemvidEncoder, MemvidChat
import os
from dotenv import load_dotenv

def get_api_key():
    
    """Get the OpenAI API key from environment variables."""
    # Create venv directory if it doesn't exist
    venv_dir = "venv"
    if not os.path.exists(venv_dir):
        os.makedirs(venv_dir)
    
    # Load environment variables from venv/.env file
    env_path = os.path.join(venv_dir, ".env")
    load_dotenv(env_path)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in venv/.env file.")
        print("The chat will operate in context-only mode without AI responses.")
        return None
    return api_key

def main():
    # 1. Initialize the encoder with default settings
    encoder = MemvidEncoder()  # Use default configuration

    # 2. Process each file in the inputs directory
    inputs_dir = "inputs"
    for filename in os.listdir(inputs_dir):
        file_path = os.path.join(inputs_dir, filename)
        if not os.path.isfile(file_path):
            continue

        # Determine file type and process accordingly
        if filename.lower().endswith('.pdf'):
            print(f"Processing PDF file: {filename}")
            encoder.add_pdf(file_path)
        else:
            print(f"Processing text file: {filename}")
            with open(file_path, 'r') as f:
                content = f.read()
                encoder.add_text(content)

    # 3. Build the video memory file and index
    video_path = "test_memory.mp4"
    index_path = "test_index.json"
    encoder.build_video(video_path, index_path)

    # 4. Initialize the chat interface
    api_key = get_api_key()
    chat = MemvidChat(
        video_file=video_path,
        index_file=index_path,
        llm_api_key=api_key,
        llm_provider="openai"
    )

    # 5. Start an interactive chat session
    print("\nStarting interactive chat session...")
    print("Type your questions about the content.")
    print("Type 'quit' or 'exit' to end the session.")
    print("Type 'clear' to clear conversation history.")
    print("Type 'stats' to see session statistics.")
    print("-" * 50)

    chat.start_session()

    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()

            # Check for exit command
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            # Check for clear command
            if user_input.lower() == 'clear':
                chat.clear_history()
                print("Conversation history cleared.")
                continue

            # Check for stats command
            if user_input.lower() == 'stats':
                stats = chat.get_stats()
                print(f"\nSession stats:")
                print(f"Messages exchanged: {stats['messages_exchanged']}")
                print(f"LLM provider: {stats['llm_provider']}")
                print(f"LLM available: {stats['llm_available']}")
                continue

            # Skip empty input
            if not user_input:
                continue

            # Get response
            print("\nAssistant: ", end="", flush=True)
            response = chat.chat(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue

if __name__ == "__main__":
    main()