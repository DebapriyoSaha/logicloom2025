import streamlit as st
from transformers import pipeline
# from dotenv import load_dotenv
import os
import re
import torch
import string
import time

# Set page title and icon
st.set_page_config(
    page_title="Headline Helper 📰", 
    page_icon="📰", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS for full dark mode styling
st.markdown(
    """
    <style>
    /* General Styling for Dark Mode */
    body {
        font-family: 'Arial', sans-serif;
        color: #f5f5f5;
        background-color: #121212;
        margin: 0;
        padding: 0;
    }
    .stApp {
        background-color: #121212;
    }
    header {
        text-align: center;
        margin-bottom: 20px;
    }
    .reportview-container {
        background: #121212;
    }
    
    /* Dark Mode Styles */
    .stApp {
        background-color: #121212;  /* Full dark background */
        color: #f5f5f5;  /* Light text color */
    }

    .stButton > button {
        background-color: #2a9d8f;  /* Button color */
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #219e80;  /* Button hover color */
    }

    /* Headline and other containers with darker background */
    .headline-box, .info-box, .error-box, .success-box {
        background-color: #333333;  /* Dark background for boxes */
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #f5f5f5;  /* Light text in boxes */
    }

    .headline-box {
        border: 1px solid #444444;  /* Light border for the boxes */
    }

    .info-box {
        background-color: #4a90e2;
        border: 1px solid #356ab2;
    }
    .error-box {
        background-color: #ff4d4d;
        border: 1px solid #c32a2a;
    }
    .success-box {
        background-color: #2e8b57;
        border: 1px solid #206c43;
    }

    /* Styling the text inputs */
    .stTextInput, .stTextArea {
        background-color: #333333;  /* Input background */
        color: #f5f5f5;  /* Input text color */
        border: 1px solid #444444;  /* Border color for input */
        padding: 10px;
    }

    .stTextInput:focus, .stTextArea:focus {
        border-color: #2a9d8f;  /* Highlight border on focus */
    }

    /* Mobile Responsive Design */
    @media (max-width: 768px) {
        body {
            font-size: 14px;
        }
        .stButton > button {
            padding: 8px 16px;
            font-size: 14px;
        }
        .headline-box, .info-box, .error-box, .success-box {
            padding: 12px;
            margin: 8px 0;
        }
        .stTextInput, .stTextArea {
            font-size: 14px;
            padding: 10px;
        }
        .stMarkdown {
            font-size: 16px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load environment variables
# load_dotenv()
# api_key = os.getenv("HF_TOKEN")
peft_model_name = "debapriyo/LogicLoom2025_BART_latest"
base_model_name = "facebook/bart-large-cnn"

# @st.cache_resource
def load_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pipe = pipeline("text2text-generation", model="debapriyo/LogicLoom2025_BART_latest")

#     # model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=api_key)
#     torch._dynamo.optimize("eager")
#     base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, torch_dtype=torch.float16)
#     print(base_model)

#     # model = PeftModel.from_pretrained(base_model, peft_model_name)
    

#     model = AutoModelForCausalLM.from_pretrained(
#             peft_model_name,
#             torch_dtype=torch.float16,
#             device_map="auto"  # Automatically map the model to available devices
#             ).to(device)

#     print(model)
#     tokenizer = AutoTokenizer.from_pretrained(peft_model_name, use_fast=True,trust_remote_code=True)
#     # model.generation_config.cache_implementation = "static"

#     # model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    return pipe

pipe = load_model()

# Preprocess the input text
def preprocess_text(text):
    text = re.sub(r'\(\w+ \d{1,2}, \d{4}.*?\)', '', text).strip()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[^\w\s,.]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text

def generate_response(input_text):

    # Perform inference
    output = pipe(input_text, max_length=60, temperature=0.7, num_beams=4, early_stopping=True, length_penalty=2, do_sample=True,)
    
    decoded_output=output[0]["generated_text"]
    return decoded_output

# Define a function to validate text input
def is_valid_plain_text(input_text):
    # Look for common programming patterns
    code_patterns = [
        r'[{}\[\]<>]',  # Braces and brackets
        r'(#|//|/\*|\*/)',  # Comments
        r'function\s*\(|var\s|let\s|const\s',  # JavaScript patterns
        r'public\s|private\s|protected\s',  # Java/other OOP keywords
    ]
    for pattern in code_patterns:
        if re.search(pattern, input_text):
            return False
    return True

# Streamlit app definition
st.title("📰 Headline Helper: Create Engaging News Headlines")
st.markdown(
    """
    Welcome to **Headline Helper**!  
    This app generates concise and click-worthy headlines for your news articles.  
    Please enter a plain text article (minimum **50 words**) in the box below to get started.
    """
)

# Input box for the news article
article = st.text_area(
    "Enter the news article (plain text only, minimum 50 words):", 
    height=300, 
    help="Paste the full news article here. Ensure it contains plain text only and is at least 50 words long."
)

# Generate the headline when the button is clicked
if st.button("✨ Generate Headline"):
    if article.strip():
        word_count = len(article.split())
        if word_count < 50:
            st.markdown(
                f'<div class="error-box">⚠️The article is too short! It contains only {word_count} words. Please enter at least 50 words.</div>',
                unsafe_allow_html=True,
            )
        elif not is_valid_plain_text(article):
            st.markdown(
                '<div class="error-box">⚠️ The input contains code or programming-related content. Please enter plain text only.</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Generating headline..."):
                start = time.perf_counter()
                
                generated_headline = generate_response(article)
                
                response_text = generated_headline.strip()
                headline_length = len(response_text.split())
                
                st.markdown(
                    f'<div class="success-box">🎉 <strong>Generated Headline:</strong> {response_text}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div> 📊 <strong>Headline Length:</strong> {headline_length} words<br>📖 <strong>Article Word Count:</strong> {word_count} words</div>',
                    unsafe_allow_html=True,
                )
                end = time.perf_counter()
                st.write(f"⏱️ Time taken to generate the headline: {end - start:.2f} seconds")
    else:
        st.markdown(
            '<div class="error-box">⚠️ Please enter a news article to generate a headline.</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")
st.markdown("📝 Developed by [Debapriyo Saha](https://www.linkedin.com/in/debapriyo-saha/)")
