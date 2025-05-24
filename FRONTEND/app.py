from flask import Flask, render_template

app = Flask(__name__, static_url_path="/static")

@app.route("/")
def index():
    # Simplemente devuelve la plantilla; toda la lógica está en JS
    return render_template("index.html")

if __name__ == "__main__":
    # Flask en 5000; tu API FastAPI en 8000
    app.run(debug=True, host="0.0.0.0", port=5000)
