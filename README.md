# Watermelon Lovers 🍉

**Watermelon Lovers** is a senior design project developed at the **University of Texas at Austin**. The project aims to create a consumer-friendly tool that evaluates the quality and ripeness of watermelons using acoustic and visual analysis techniques. By leveraging machine learning and signal processing, this tool will help users make informed decisions while shopping for watermelons at grocery stores and markets.

This repository contains the codebase for the project, including a frontend for user interaction and a backend (currently under development) for data processing and evaluation.

---

## Project Structure

- **frontend/**: Contains the user-facing application, built using Python's Flask framework.
  - `app.py`: The main entry point for the frontend server.
  - `templates/`: HTML templates for the application's web pages.
    - `home`: The landing page where users can upload their data.
    - `results`: Displays the evaluation of watermelon quality and ripeness.
  - `uploads/`: Stores user-uploaded files (e.g., images or acoustic data).

- **backend/**: Reserved for future development, including the core machine learning and data processing logic.

- - **watermelon_acoustic scripts/**:
  - `main.py`: Orchestrates the entire pipeline. Configures paths, calls each stage (preprocessing, standardization, noise reduction, model training, testing), and ensures a smooth flow from raw data to final evaluation.
  - `wav_file_converter.py`: Converts `.m4a` files to `.wav` (and handles other basic format conversions) so all audio data is in a consistent `.wav` format.
  - `standardize_wav.py`: Ensures each `.wav` file meets uniform specs (sample rate, mono channel, 16-bit PCM, exact duration, etc.). Trims or pads audio as needed.
  - `noise_reduction.py`: Applies one or more noise reduction or filtering techniques (e.g. low-pass filter, gain adjustments) to improve audio clarity before model training.
  - `model_training.py`: Loads the standardized/noise-reduced `.wav` files, parses labels from filenames, splits data into train/val/test, and trains a machine learning model (e.g., an LSTM) to predict watermelon sweetness (Brix).
  - `model_testing.py`: Loads trained models and tests them against a specified dataset (e.g., a hold-out test set), computing metrics like MAE, MSE, and R². Can also generate visual charts (bar charts, scatter plots, etc.) comparing model performance.


---

## Getting Started

### Prerequisites

- **Python 3.9 or later**
- Required Python packages (install via pip):
  ```bash
  pip install flask
