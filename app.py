

from flask import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

data = pd.read_csv("conv.csv")

app = Flask(__name__)
app.secret_key = "monika_secret"

@app.route("/", methods=["GET", "POST"])
def home():

    if "chat" not in session:
        session["chat"] = []

    if request.method == "POST":
        user_msg = request.form["qts"].strip().lower()

        texts = [user_msg] + data["question"].str.lower().tolist()
        cv = CountVectorizer()
        vector = cv.fit_transform(texts)
        cs = cosine_similarity(vector)
        score = cs[0][1:]

        temp_data = data.copy()
        temp_data["score"] = score
        result = temp_data.sort_values(by="score", ascending=False)

        if result.iloc[0]["score"] < 0.1:
            bot_reply = "Hmm. I’m not fully sure about that yet. Try asking in a different way 😊"
        else:
            bot_reply = result.iloc[0]["answer"]

        session["chat"].append(("user", user_msg))

        time.sleep(0.6)

        session["chat"].append(("bot", bot_reply))
        session.modified = True

    return render_template("home.html", chat=session["chat"])

if __name__ == "__main__":
    app.run(debug=True)
