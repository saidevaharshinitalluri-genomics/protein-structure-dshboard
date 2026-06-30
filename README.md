
# Protein Secondary Structure Prediction (PSSP) Dashboard 🧬

An interactive deep learning web application designed to predict protein secondary structures (Alpha Helix, Beta Strand, and Coil) directly from amino acid sequences. 

## Key Features
* Predicts secondary structures from raw sequences using a Bidirectional GRU model.
* Achieves an 89% prediction accuracy rate.
* Processes sequences using 63-dimensional physicochemical and positional feature vectors.
* Integrates a multi-model ensemble utilizing the classic GOR IV algorithm as a fallback.
* Provides an interactive Gradio web interface with 3D structure visualization and confidence heatmaps.
* Supports batch processing of multiple sequences via FASTA file uploads.

## Tech Stack
* **Machine Learning:** PyTorch, Scikit-learn
* **Frontend/Dashboard:** Gradio
* **Data Visualization:** Plotly, Matplotlib
* **Data Processing:** NumPy, Pandas

## Repository Contents
* `app.py`: The Gradio web dashboard script.
* `best_gru_fusion_cb513.pth`: The trained Bidirectional GRU model weights.
* `requirements.txt`: Project dependencies.

## How to Run Locally
1. Clone this repository to your local machine.
2. Install the required dependencies by running: `pip install -r requirements.txt`
3. Launch the dashboard by running: `python app.py`
4. Open the provided local URL in your web browser.
