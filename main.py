import streamlit as st

from model import NlpModel

nlp_model = NlpModel()
st.title("Understanding Siri Project")
st.header('Predict your voice command intents with transformers! :sunglasses:')

user_input = st.text_input("Enter voice command")

if st.button('Predict!'):
    with st.spinner('Generating ...'):
        st.text("BERT Tokens: ")
        bert_tokens = nlp_model.return_bert_tokens(user_input)
        st.text(bert_tokens)
        st.text("BIO Tags: ")
        intent, slots = nlp_model.show_predictions(user_input)
        st.text(intent)
        st.text(slots)
        st.text("Overall Predictions: ")
        output = nlp_model.nlu(user_input)
        st.text(output)
