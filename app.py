import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# DOWNLOAD NLTK DATA
# =========================
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# =========================
# LOAD MODEL
# =========================
ps = PorterStemmer()

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# =========================
# TEXT PREPROCESSING
# =========================
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    # Keep only letters and numbers
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📌 Project Information")

st.sidebar.markdown("""
## SMS Spam Detection

This project predicts whether an SMS is **Spam** or **Ham** using Machine Learning.

### Technologies Used
- Python
- Streamlit
- Scikit-Learn
- TF-IDF
- NLTK

### Model
Multinomial Naive Bayes

### Dataset
SMS Spam Collection Dataset

---

👨‍💻 Developed by

**K. Rakesh Reddy**
""")

# =========================
# HEADER
# =========================
st.markdown(
    """
# 📩 SMS Spam Detection System

Detect whether an SMS is **Spam** or **Ham** using Machine Learning.
"""
)

st.write("---")

# =========================
# INPUT
# =========================
input_sms = st.text_area(
    "✍ Enter your SMS Message",
    height=180
)

# =========================
# MESSAGE STATISTICS
# =========================
if input_sms:

    st.subheader("📊 Message Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Characters", len(input_sms))
    col2.metric("Words", len(input_sms.split()))
    col3.metric("Sentences", len([s for s in input_sms.split('.') if s.strip()]))

st.write("")

# =========================
# PREDICT BUTTON
# =========================
if st.button("🚀 Predict"):

    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])

    prediction = model.predict(vector_input)

    probability = model.predict_proba(vector_input)

    confidence = probability.max() * 100

    st.write("---")

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.error("🚨 SPAM MESSAGE")

        st.progress(int(confidence))

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.warning(
            "This message appears to be Spam. "
            "Avoid clicking unknown links or sharing personal information."
        )

    else:

        st.success("✅ HAM MESSAGE")

        st.progress(int(confidence))

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.info(
            "This message appears to be a legitimate message."
        )

# =========================
# FOOTER
# =========================
st.write("---")

st.caption(
    "Developed by K. Rakesh Reddy | SMS Spam Detection using Machine Learning"
)