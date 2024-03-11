import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import NotFittedError

# Simplified model training and testing
def train_and_predict(user_input, x, y, model):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)

    model.fit(xv_train, y_train)

    try:
        new_x_test = pd.Series([user_input])
        new_xv_test = vectorization.transform(new_x_test)

        pred = model.predict(new_xv_test)

        st.write(f"{model.__class__.__name__} Prediction:", output_label(pred[0]))

    except NotFittedError as e:
        st.error("Error: Vectorizer not fitted. Please ensure the vectorizer is fitted before prediction.")

def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

# Streamlit UI
def main():
    st.title("Fake News Detection App")

    # Load your data here
    dataframe_fake = pd.read_csv("Fake.csv")
    dataframe_true = pd.read_csv("True.csv")
    dataframe_fake["class"] = 0
    dataframe_true["class"] = 1
    dataframe_merge = pd.concat([dataframe_fake, dataframe_true], axis=0)
    dataframe = dataframe_merge.drop(["title", "subject", "date"], axis=1)

    def wordopt(t):
        # Include your text preprocessing logic here
        return t

    dataframe["text"] = dataframe["text"].apply(wordopt)

    # Define x and y
    x = dataframe["text"]
    y = dataframe["class"]

    user_input = st.text_area("Enter the news text:")

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=0),
        "Random Forest": RandomForestClassifier(random_state=0)
    }

    selected_model = st.selectbox("Select a model:", list(models.keys()))

    if st.button("Predict"):
        st.write("Predictions:")
        train_and_predict(user_input, x, y, models[selected_model])  # Pass x, y, and selected model to the function

if __name__ == '__main__':
    main()
