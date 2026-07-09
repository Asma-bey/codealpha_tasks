import streamlit as st
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import string
import random
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ FAQ DATA ------------------
faqs = [
    ("How can I focus while studying?", "Try the Pomodoro technique: 25 min study, 5 min break."),
    ("How do I stop procrastinating?", "Start small and remove distractions like your phone."),
    ("What is the best time to study?", "Morning is great, but consistency matters more."),
    ("How many hours should I study?", "Focus on quality: 2-4 productive hours is enough."),
    ("How can I remember things better?", "Use active recall and spaced repetition."),
    ("What should I do before exams?", "Revise and practice past papers."),
    ("How to stay motivated?", "Set goals and reward yourself."),
    ("Is studying at night bad?", "No, as long as you sleep enough.")
]

quotes = [
    "💪 Push yourself, because no one else will do it for you.",
    "📚 Small progress is still progress.",
    "🔥 Discipline beats motivation.",
    "🚀 Success starts with consistency.",
    "🎯 Focus on your goal, not distractions."
]

# ------------------ PREPROCESS ------------------
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w.isalnum() and w not in stop_words]
    return " ".join(tokens)

questions = [preprocess(q) for q, a in faqs]
answers = [a for q, a in faqs]

# ------------------ TF-IDF ------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# ------------------ CHATBOT FUNCTION ------------------
def get_response(user_input):
    user_input = preprocess(user_input)
    user_vec = vectorizer.transform([user_input])

    similarities = cosine_similarity(user_vec, X)
    best_index = similarities.argmax()
    best_score = similarities[0][best_index]

    if best_score < 0.3:
        return random.choice([
            "🤔 I’m not sure about that.",
            "📖 Try asking about studying tips!",
            "😅 I don’t have that answer yet."
        ])

    return "📚 " + answers[best_index]

# ------------------ UI ------------------
st.title("🎓 Study Assistant Chatbot")

st.write("Ask me anything about studying!")

user_input = st.text_input("You:")

if user_input:
    response = get_response(user_input)
    st.write("Bot:", response)

if st.button("✨ Motivate Me"):
    st.success(random.choice(quotes))