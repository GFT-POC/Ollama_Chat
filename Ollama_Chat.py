import streamlit as st
import ollama
from transformers import GPT2TokenizerFast

# Initialize the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Set the title of the Streamlit app
st.title("Ollama Chatbot")

# Model options with corresponding max token limits, parameter counts, and descriptions
MODEL_DETAILS = {
    "llama3.2:1b": {
        "max_tokens": 128000,
        "parameters": "1 billion parameters",
        "description": "Fast and efficient, suited for simple, quick-response tasks."
    },
    "llama3.2:latest": {
        "max_tokens": 128000,
        "parameters": "3 billion parameters",
        "description": "Enhanced reasoning, ideal for moderately complex applications."
    },
    "llama3.1:latest": {
        "max_tokens": 128000,
        "parameters": "8 billion parameters",
        "description": "High-capacity, suitable for in-depth analyses and detailed responses."
    },
    "gemma2:2b": {
        "max_tokens": 8196,
        "parameters": "2 billion parameters",
        "description": "Balanced for speed and accuracy, handles general queries well."
    },
    "gemma2:9b": {
        "max_tokens": 8196,
        "parameters": "9 billion parameters",
        "description": "Increased accuracy, excellent for nuanced, complex question answering."
    },
    "nemotron-mini": {
        "max_tokens": 4096,
        "parameters": "4 billion parameters",
        "description": "Compact model, optimal for concise contexts with reliable accuracy."
    },
    "mistral": {
        "max_tokens": 32768,
        "parameters": "7 billion parameters",
        "description": "Balanced for speed and complexity, suitable for mid-level tasks."
    },
    "mistral-nemo": {
        "max_tokens": 128000,
        "parameters": "12 billion parameters",
        "description": "High-capacity, tailored for complex, multi-turn conversations."
    },
    "smollm2": {
        "max_tokens": 2048,
        "parameters": "1.7 billion parameters",
        "description": "Efficient and compact, best for brief, straightforward tasks."
    },
    "llama2-uncensored:latest": {
        "max_tokens": 4096,
        "parameters": "7 billion parameters",
        "description": "Open-ended, ideal for exploratory, unrestricted dialogues."
    }
}

# Dropdown for model selection
selected_model = st.selectbox("Choose a model", options=list(MODEL_DETAILS.keys()))
model_info = MODEL_DETAILS[selected_model]
MAX_TOKENS = model_info["max_tokens"]
PARAMETERS = model_info["parameters"]
DESCRIPTION = model_info["description"]

# Constants for word and page calculation
TOKENS_PER_WORD = 1.3653
WORDS_PER_PAGE = 500

# Calculate approximate words and pages
approx_words = round(MAX_TOKENS / TOKENS_PER_WORD)
approx_pages = round(approx_words / WORDS_PER_PAGE)

# Display the context window and parameter information for the selected model with formatted numbers
st.write(f"Model: {selected_model}")
st.write(f"- Parameters: {PARAMETERS}")
st.write(f"- Description: {DESCRIPTION}")
st.write(f"- Context window (tokens): {MAX_TOKENS:,}")
st.write(f"- Approximate words: {approx_words:,}")
st.write(f"- Approximate pages: {approx_pages:,}")

# Initialize session state to store chat history and input
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "first_message" not in st.session_state:
    st.session_state.first_message = True

# Function to count tokens in a message
def count_tokens(message):
    return len(tokenizer.encode(message))

# Function to truncate messages to fit within the context window
def truncate_messages(messages, max_tokens):
    total_tokens = 0
    truncated_messages = []
    for message in reversed(messages):
        message_tokens = count_tokens(message["content"])
        if total_tokens + message_tokens > max_tokens:
            break
        truncated_messages.insert(0, message)
        total_tokens += message_tokens
    return truncated_messages

# Function to handle user input submission
def submit():
    user_message = st.session_state.user_input
    if user_message:
        # Append user's message to chat history
        st.session_state.messages.append({"role": "user", "content": user_message})

        # Prepare the message payload for the model including truncated history
        messages = truncate_messages(st.session_state.messages, MAX_TOKENS)

        # Send the message to the model
        try:
            response = ollama.chat(model=selected_model, messages=messages)
            bot_response = response["message"]["content"]
        except ollama.ResponseError as e:
            bot_response = f"Error: {e.error}"

        # Append bot's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

        # Clear the input field after submission
        st.session_state.user_input = ""

# Display chat history
for message in st.session_state.messages:
    role = "User" if message["role"] == "user" else "Assistant"
    st.write(f"{role}: {message['content']}")

# Create a form for user input
with st.form(key="user_input_form", clear_on_submit=True):
    st.text_input("You:", key="user_input")
    submit_button = st.form_submit_button(label="Send", on_click=submit)
