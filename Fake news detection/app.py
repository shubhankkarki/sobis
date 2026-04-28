import streamlit as st
import pickle
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
st.title("Fake News Detection using NLP")
news = st.text_area("Enter News Article")
if st.button("Predict"):
    news_vec = vectorizer.transform([news])
    prediction = model.predict(news_vec)[0]
    if prediction == 0:
        st.error("The news is Fake")
    else:
        st.success("The news is True")