from flask import Flask, render_template
from blueprints.pushups import pushups_bp
from blueprints.biceps import biceps_bp
from blueprints.squats import squats_bp
from blueprints.diet import diet_bp

app = Flask(__name__, template_folder='templates')

# Registering blueprints
app.register_blueprint(pushups_bp, url_prefix="/blueprints/pushups")
app.register_blueprint(biceps_bp, url_prefix="/blueprints/biceps")
app.register_blueprint(squats_bp, url_prefix="/blueprints/squats")
app.register_blueprint(diet_bp, url_prefix="/blueprints/diet")


@app.route('/')
def home():
    return render_template('home.html')  # Render the combined HTML file


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=True)