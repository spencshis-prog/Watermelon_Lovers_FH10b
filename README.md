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

---

## Getting Started

### Prerequisites

- **Python 3.9 or later**
- Required Python packages (install via pip):
  ```bash
  pip install flask
