from flask import Flask, render_template, request
from function import globalFunction
app = Flask(__name__)

@app.route('/', methods=["POST", "GET"])
def home():
    predict = "-"
    if request.method=="POST":
        InputUser = request.form["pred"]
        ppr = globalFunction().data_processing(InputUser)
        ttsr = globalFunction().tts(ppr)
        lmr = globalFunction().predict(ttsr)
        return render_template('index.html', predict=lmr)

    else: return render_template('index.html', predict=predict) 

if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=8080)
