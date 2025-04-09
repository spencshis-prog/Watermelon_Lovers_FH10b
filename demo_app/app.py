# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
import os
import json

app = Flask(__name__)
app.secret_key = "CHANGE_ME_TO_A_STRONG_SECRET_KEY"

# Home Page: explains the project and links to results.
@app.route('/')
def home():
    return render_template('home.html')

# Results Page: List all watermelon_data folders.
@app.route('/results')
def results():
    watermelons_dir = os.path.join(os.getcwd(), "watermelon_data")
    watermelons = []
    if os.path.exists(watermelons_dir):
        for entry in os.listdir(watermelons_dir):
            entry_path = os.path.join(watermelons_dir, entry)
            if os.path.isdir(entry_path):
                watermelons.append(entry)
    return render_template('results.html', watermelons=watermelons)

# Helper route to serve files from a specific watermelon folder.
@app.route('/watermelon_data/<wm_folder>/<filename>')
def serve_watermelon_file(wm_folder, filename):
    data_dir = os.path.join(os.getcwd(), "watermelon_data", wm_folder)
    return send_from_directory(data_dir, filename)

# Unique Watermelon Page: GET shows the FFT plot and forms; POST stores real Brix or calls predict.py.
@app.route('/watermelon/<wm_folder>', methods=['GET', 'POST'])
def watermelon_page(wm_folder):
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "watermelon_data", wm_folder)
    if not os.path.exists(data_dir):
        flash("Watermelon data not found.")
        return redirect(url_for('results'))
    
    # Use the fft_plot.png if available; otherwise, fall back to a placeholder image.
    fft_plot_path = os.path.join(data_dir, "fft_plot.png")
    if os.path.exists(fft_plot_path):
        fft_plot_url = url_for("serve_watermelon_file", wm_folder=wm_folder, filename="fft_plot.png")
    else:
        fft_plot_url = url_for("static", filename="placeholder.jpg")
    
    fft_data_path = os.path.join(data_dir, "fft_data.json")
    fft_data = {}
    if os.path.exists(fft_data_path):
        with open(fft_data_path, 'r') as f:
            fft_data = json.load(f)
    
    message = None
    if request.method == "POST":
        action = request.form.get("action")
        if action == "store_real":
            # The user inputs the real Brix value.
            real_brix = request.form.get("real_brix")
            if real_brix:
                fft_data["real_brix"] = real_brix
                message = f"Real Brix ({real_brix}) stored successfully."
            else:
                message = "Please provide a valid real Brix value."
        elif action == "predict":
            # Call the prediction function from predict.py.
            wav_file = os.path.join(data_dir, "watermelon.wav")
            model_path = os.path.join(os.getcwd(), "regressor_model.pkl")
            try:
                from predict import predict_brix
                predicted_brix = predict_brix(wav_file, model_path)
                fft_data["predicted_brix"] = predicted_brix
                message = f"Predicted Brix: {predicted_brix}"
            except Exception as e:
                message = f"Prediction error: {e}"
        
        # Update the fft_data.json with the new information.
        with open(fft_data_path, 'w') as f:
            json.dump(fft_data, f, indent=4)
        flash(message)
    
    return render_template("watermelon.html", wm_folder=wm_folder, fft_plot_url=fft_plot_url, fft_data=fft_data)

if __name__ == '__main__':
    app.run(debug=True)
