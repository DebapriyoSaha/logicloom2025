import streamlit as st
from transformers import AutoTokenizer, AutoModel
from peft import AutoPeftModelForCausalLM
import os
import re
import string
from dotenv import load_dotenv
import torch  # Import PyTorch to check for device availability
import time
from transformers import BitsAndBytesConfig


# bnb_config = BitsAndBytesConfig(load_in_8bit=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,  # Use mixed precision (float16)
    bnb_4bit_use_double_quant=True,  # Double quantization for memory efficiency
    bnb_4bit_quant_type="nf4"  # Use NormalFloat4 quantization (better for fine-tuning)
)
# Load environment variables
load_dotenv()
# api_key = os.getenv("HUGGING_FACE_API_KEY")
api_key = "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"
model_name= "debapriyo/LogicRoom25_AlpacaFormat_Gemma_2_9B_4epoch_latest"

# Load the Hugging Face model and tokenizer
@st.cache_resource
def load_model():
    model = AutoPeftModelForCausalLM.from_pretrained(model_name, token=api_key)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_key)
    return tokenizer, model

tokenizer, model = load_model()
#### Preprocess the input text
# Check if GPU is available and move model to the correct device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def preprocess_text(text):

  # Remove patterns like "(Oct 5, 2011 12:11 PM CDT)"
  text = re.sub(r'\(\w+ \d{1,2}, \d{4}.*?\)', '', text).strip()  
    
  # Remove URLs (http, https)
  text = re.sub(r'http\S+', '', text)

  # Remove non-printable or non-ASCII characters if necessary
  text = re.sub(r'[^\x00-\x7F]+', '', text)

  # Remove unecessary alphabets  
  text = re.sub(r'[^\w\s,.]', '', text)

  # Remove punctuation
  text = text.translate(str.maketrans('', '', string.punctuation))

  # Convert to lowercase
  text = text.lower()

  return text


def prepare_input_text(text):

    input_text = f"Analyze the news content to generate creating engaging, concise, and click-worthy titles that resonate with readers and drive traffic. Ensure your answer is based on the information from the News.\n\n News: {text}."

    return input_text

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

            ### Instruction:
            {}

            ### Input:
            {}

            ### Response:
            {}"""

def generate_response(input_text):
    instruction = "You are an expert in generating concise, creative and contextually relevant caption."
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            instruction,
            input_text,  # Leave input blank if not needed
            ""  # Leave output blank for generation
        )
    ], return_tensors = "pt").to("cuda")
    
    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    
    generated = model.generate(input_ids = inputs.input_ids, attention_mask = inputs.attention_mask,
                       streamer = text_streamer, max_new_tokens = 64, pad_token_id = tokenizer.eos_token_id)
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)                


# Set page title and icon
st.set_page_config(
    page_title="Headline Helper", 
    page_icon="📰", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Define the Streamlit app
st.title("Headline Helper")
st.markdown("This app generates a headline for a given news article.")

# Input box for the news article
article = st.text_area("Enter the news article:", height=300)

# Generate the headline when the button is clicked
if st.button("Generate Headline"):
    if article.strip():
        with st.spinner("Generating headline..."):

            start = time.perf_counter()
            # Prepare the input for the model
            preprocessed_text = prepare_input_text(article)
            
            generated_headline = generate_response(preprocessed_text)

            response_start = "### Response:"
            response_text = generated_headline.split(response_start)[-1].strip()
            
            st.success(f"Generated Headline: {response_text}")
            # st.write(response_text)

            end = time.perf_counter()
            st.markdown("---")
            st.write(f"Time taken to generate the headline: {end - start:.2f} seconds")
    else:
        st.error("Please enter a news article to generate a headline.")

    st.markdown("---")
    # st.markdown("Model: [Fine Tuned Gemma-2-9b](https://huggingface.co/debapriyo/LogicRoom25_AlpacaFormat_Gemma_2_9B_4epoch_latest)")