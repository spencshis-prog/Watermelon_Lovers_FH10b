from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
import os
import numpy as np

app = Flask(__name__)

# Define the folder to save uploads
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Serve files from the upload folder
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Homepage Route
@app.route('/')
def home():
    return render_template('home.html')

# Upload Route
@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(url_for('home'))

    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('home'))

    if not allowed_file(file.filename):
        flash('File type not allowed. Please upload an image (jpg, png, jpeg, gif).')
        return redirect(url_for('home'))

    # Save the uploaded file
    filename = file.filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    flash(f"File {filename} uploaded successfully!")
    return redirect(url_for('results'))

# Results Page Route
@app.route('/results')
def results():
    # Get a list of uploaded files
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    file_paths = [url_for('uploaded_file', filename=f) for f in files]

    # Placeholder analysis results
    results = {file: f"Analysis result for {os.path.basename(file)}" for file in file_paths}

    return render_template('results.html', file_paths=file_paths, results=results)

# API endpoint to serve FFT plot
@app.route('/fft_plot/<int:watermelon_id>', methods=['GET'])
def get_fft_plot(watermelon_id):
    plot_path = f"../backend/watermelon_data/watermelon_{watermelon_id}/fft_plot.png"
    if os.path.exists(plot_path):
        return send_from_directory(os.path.dirname(plot_path), os.path.basename(plot_path))
    else:
        return jsonify({"error": "Plot not found"}), 404

# API endpoint to serve FFT data
@app.route('/fft_data/<int:watermelon_id>', methods=['GET'])
def get_fft_data(watermelon_id):
    data_path = f"../backend/watermelon_data/watermelon_{watermelon_id}/fft_data.npy"
    if os.path.exists(data_path):
        fft_data = np.load(data_path).tolist()
        return jsonify(fft_data)
    else:
        return jsonify({"error": "Data not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)