#!/usr/bin/env python
# coding: utf-8

# In[1]:


from faster_whisper import WhisperModel
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
import gradio as gr
import sentencepiece
import openai
from dotenv import load_dotenv
import os
import speech_recognition as sr

load_dotenv()


# In[20]:


# Retrieve the OpenAI API key from the environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Ensure that the API key is loaded
if not openai.api_key:
    raise ValueError("OpenAI API key is not set in the environment variables.")


# ## Faster Whisper for Audio ##
# 

# In[4]:


### 1. Faster Whisper for Audio Transcription ###
def transcribe_audio(audio_file):
    model = WhisperModel("small", device="cpu")  # Use "cuda" if you have a GPU
    segments, _ = model.transcribe(audio_file, language="es")
    transcription = " ".join([segment.text for segment in segments])
    return transcription


# ## 2. Translation Pipelines ###

# In[5]:


# 2. Translation Pipelines
translator_es_en = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en", device="cpu")
translator_en_es = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es", device="cpu")


# In[6]:


# translate from spanish to english
def translate_to_english(text):
    translated = translator_es_en(text, max_length=1024)
    return translated[0]['translation_text']

# Translate from english to spanish
def translate_to_spanish(text):
    translated = translator_en_es(text, max_length=1024)
    return translated[0]['translation_text']


# ## 3. RAG Setup (Using LangChain and FAISS)
# 

# In[ ]:


# Create the vector store and embedding.
def create_vector_db(resume_file):
    loader = TextLoader(resume_file)
    documents = loader.load()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Generate the response wtih gpt-4o-mini
def generate_response(question, vectorstore):
    instructions = "Please answer the following question based on the resume, and be creative with your responses."
    full_query = instructions + "\nQuestion: " + question
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature=0.7)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa_chain.invoke(full_query)

# Load the RAG Knowledge Base
vectorstore = create_vector_db("Resume.txt")


# ## 4. Complete Pipeline

# In[ ]:


def process_pipeline(audio_file):
    # Step 1: Transcribe audio (Spanish)
    spanish_text = transcribe_audio(audio_file)
    
    # Step 2: Translate Spanish transcription to English
    english_query = translate_to_english(spanish_text)
    
    # Step 3: Query the RAG system in English
    english_response = generate_response(english_query, vectorstore)

    # Make sure the response is a string, and extract it if necessary
    if isinstance(english_response, dict):
        english_response = english_response.get('result', '')  # Extract the text if it's a dict
    
    # Step 4: Translate the English response back to Spanish
    spanish_response = translate_to_spanish(english_response)
    
    return spanish_text, english_response, spanish_response


def check_pronunciation(audio_file, expected_text):
    # Use a speech recognition library to transcribe the audio
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        # Transcribe the audio to text
        transcribed_text = recognizer.recognize_google(audio, language="es-ES")
        # Compare transcribed text with expected text
        feedback = "Correct pronunciation!" if transcribed_text == expected_text else "Try again."
    except sr.UnknownValueError:
        feedback = "Could not understand audio."
    except sr.RequestError as e:
        feedback = f"Could not request results; {e}"
    return feedback


def combined_interface(audio_file):
    # Step 1: Process the question and get responses
    spanish_text, english_response, spanish_response = process_pipeline(audio_file)
    
    # Step 2: Prompt for pronunciation feedback
    feedback = "Please record your pronunciation of the Spanish response."
    
    return spanish_text, english_response, spanish_response, feedback


import gradio as gr


with gr.Blocks() as demo:
    gr.Markdown("# Spanish-English Q&A System with Pronunciation Feedback")
    gr.Markdown("Ask questions in Spanish via audio, receive responses in both Spanish and English, and get pronunciation feedback.")
    
    # Step 1: Question and Response
    with gr.Row():
        audio_input = gr.Audio(sources="microphone", type="filepath", label="Ask your question in Spanish")
        question_button = gr.Button("Submit Question")
    
    spanish_output = gr.Textbox(label="Transcribed Spanish Text")
    english_output = gr.Textbox(label="English Response")
    spanish_response_output = gr.Textbox(label="Spanish Response")

    question_button.click(combined_interface, inputs=audio_input, outputs=[spanish_output, english_output, spanish_response_output])

    # Step 2: Pronunciation Feedback
    with gr.Row():
        pronunciation_input = gr.Audio(sources="microphone", type="filepath", label="Record your pronunciation of the Spanish response")
        pronunciation_button = gr.Button("Submit Pronunciation")
    
    pronunciation_feedback_output = gr.Textbox(label="Pronunciation Feedback")

    pronunciation_button.click(check_pronunciation, inputs=[pronunciation_input, spanish_response_output], outputs=pronunciation_feedback_output)


demo.launch(share=True)
