import streamlit as st
from transformers import pipeline
import torch

# Load the pre-trained model and tokenizer
model_name = "distilbert-base-uncased-distilled-squad"
qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)

# Streamlit UI
st.title("Quantum AI Question Answering")

question = st.text_input("Enter your question:")
context = st.text_area("Enter the context:")

if st.button("Ask"):
    if question and context:
        try:
            inputs = {
                "question": question,
                "context": context
            }
            # Run the question-answering pipeline
            result = qa_pipeline(question=inputs['question'], context=inputs['context'])
            answer = result['answer']
            st.write("Answer:", answer)
        except KeyError as e:
            st.error(f"KeyError: {str(e)}. Please check the inputs and try again.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter both a question and context.")
