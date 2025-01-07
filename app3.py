import os
import re
import string
import streamlit as st
from dotenv import load_dotenv
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

torch.cuda.empty_cache()

# Release unoccupied memory
torch.cuda.memory_summary(device=None, abbreviated=False)

# Load environment variables
load_dotenv()
api_key = os.getenv("HUGGING_FACE_API_KEY")
model_name = "debapriyo/LogicRoom25_AlpacaFormat_Gemma_2_9B_4epoch_latest"

# Load the Hugging Face model and tokenizer
@st.cache_resource
def load_model():

    max_seq_length = 512
    dtype = "float16"
    load_in_4bit = True

    # Check and print device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")  # This will print the device being used

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        # dtype=dtype,
        # load_in_4bit=load_in_4bit,
        device_map="cuda",
        token=api_key,
    )
    FastLanguageModel.for_inference(model)

    model.to("cuda")

    return tokenizer, model
print("Check")
tokenizer, model = load_model()

#### Preprocess the input text
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
        ], return_tensors="pt").to("cuda")
    
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    generated = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                       streamer=text_streamer, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)

# Define the Streamlit app
st.title("Headline Helper")
st.markdown("This app generates a headline for a given news article.")

# Input box for the news article
article = st.text_area("Enter the news article:", height=300)

# Generate the headline when the button is clicked
if st.button("Generate Headline"):
    if article.strip():
        with st.spinner("Generating headline..."):
            # Prepare the input for the model
            preprocessed_text = prepare_input_text(article)
            
            generated_headline = generate_response(preprocessed_text)

            response_start = "### Response:"
            response_text = generated_headline.split(response_start)[-1].strip()
            
            st.success("Generated Headline:")
            st.write(response_text)
    else:
        st.error("Please enter a news article to generate a headline.")

st.markdown("---------------")
st.markdown("Model: [Fine Tuned Gemma-29b](https://huggingface.co/debapriyo/LogicRoom25_AlpacaFormat_Gemma_2_9B_4epoch_latest)")