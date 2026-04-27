from flask import Flask, render_template, request
import model_logic

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()

    result, prob, summary, reasons = model_logic.predict_output(form_data)

    return render_template("index.html",
                           prediction_text=f"Prediction: {result}",
                           probability=f"Confidence: {prob*100:.2f}%",
                           summary=summary,
                           reasons=reasons)

if __name__ == "__main__":
    app.run(debug=True)