import streamlit as st
import pickle
import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

ps = PorterStemmer()
def transfrom_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    x = []
    for i in text:
        if i.isalnum():
            x.append(i)

    text = x[:]
    x.clear()
    for i in text:
        if i not in string.punctuation and i not in stopwords.words('english'):
            x.append(i)


    text = x[:]
    x.clear()
    for i in text:
        x.append(ps.stem(i))
    return " ".join(x)
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SMS-Spam Classifier")

input_sms = st.text_area('Enter the message')
if st.button('Predict'):

# pre-process
    transform_sms = transfrom_text(input_sms)

#vectorize
    vector_input = tfidf.transform([transform_sms])

#predict
    result = model.predict(vector_input)[0]

#display
    if result == 1:
        st.header('SPAM')
    else:
        st.header('NOT SPAM')
