"""
Protein Secondary Structure Prediction - Gradio Interface
Fixed with Chou-Fasman fallback for realistic predictions
"""

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import re

# ==================== Model Definition ====================
class GRU_SS_Model(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, num_layers=2, num_classes=3, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers>1 else 0.0
        )
        self.classifier = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        logits = self.classifier(out)
        return logits

# ==================== Feature Engineering ====================
AA_LIST = ['A','R','N','D','C','Q','E','G','H','I',
           'L','K','M','F','P','S','T','W','Y','V']

# Enhanced physicochemical properties
AA_PROPERTIES = {
    'A':[1.8,89.09,6.00,0,8.1,67],   'R':[-4.5,174.20,10.76,1,10.5,148],
    'N':[-3.5,132.12,5.41,0,11.6,96], 'D':[-3.5,133.10,2.77,-1,13.0,91],
    'C':[2.5,121.15,5.07,0,5.5,86],   'Q':[-3.5,146.15,5.65,0,10.5,114],
    'E':[-3.5,147.13,3.22,-1,12.3,109],'G':[-0.4,75.07,5.97,0,9.0,48],
    'H':[-3.2,155.16,7.59,0,10.4,118],'I':[4.5,131.17,6.02,0,5.2,124],
    'L':[3.8,131.17,5.98,0,4.9,124],  'K':[-3.9,146.19,9.74,1,11.3,135],
    'M':[1.9,149.21,5.74,0,5.7,124],  'F':[2.8,165.19,5.48,0,5.2,135],
    'P':[-1.6,115.13,6.30,0,8.0,90],  'S':[-0.8,105.09,5.68,0,9.2,73],
    'T':[-0.7,119.12,5.60,0,8.6,93],  'W':[-0.9,204.23,5.89,0,5.4,163],
    'Y':[-1.3,181.19,5.66,0,6.2,141], 'V':[4.2,117.15,5.96,0,5.9,105]
}

# Chou-Fasman propensity parameters (scientifically validated)
HELIX_PROPENSITY = {
    'E':1.53, 'A':1.45, 'L':1.34, 'M':1.20, 'Q':1.17, 'K':1.07, 'R':1.21,
    'H':1.24, 'V':1.14, 'I':1.00, 'Y':0.61, 'C':0.77, 'W':1.14, 'F':1.12,
    'T':0.82, 'G':0.53, 'N':0.73, 'P':0.59, 'S':0.79, 'D':0.98
}

SHEET_PROPENSITY = {
    'M':1.20, 'V':1.65, 'I':1.60, 'C':1.30, 'Y':1.29, 'F':1.28, 'Q':1.23,
    'L':1.22, 'T':1.20, 'W':1.19, 'A':0.97, 'R':0.90, 'G':0.81, 'D':0.80,
    'K':0.74, 'S':0.72, 'H':0.71, 'N':0.65, 'P':0.62, 'E':0.26
}

COIL_PROPENSITY = {
    'N':1.42, 'G':1.64, 'P':1.91, 'D':1.24, 'S':1.21, 'C':1.00, 'H':0.71,
    'K':1.07, 'Q':0.98, 'E':0.99, 'T':0.90, 'R':0.88, 'A':0.66, 'Y':1.05,
    'W':0.65, 'F':0.59, 'M':0.60, 'L':0.57, 'I':0.47, 'V':0.50
}

def chou_fasman_predict(sequence):
    """
    Enhanced Chou-Fasman algorithm for secondary structure prediction
    Returns predicted structure and confidence scores
    """
    n = len(sequence)
    helix_scores = np.zeros(n)
    sheet_scores = np.zeros(n)
    coil_scores = np.zeros(n)

    # Calculate propensity scores
    for i, aa in enumerate(sequence):
        helix_scores[i] = HELIX_PROPENSITY.get(aa, 1.0)
        sheet_scores[i] = SHEET_PROPENSITY.get(aa, 1.0)
        coil_scores[i] = COIL_PROPENSITY.get(aa, 1.0)

    # Apply window smoothing (window size = 6 for helix, 5 for sheet)
    window_h = 6
    window_e = 5

    smoothed_helix = np.zeros(n)
    smoothed_sheet = np.zeros(n)
    smoothed_coil = np.zeros(n)

    for i in range(n):
        # Helix window
        h_start = max(0, i - window_h // 2)
        h_end = min(n, i + window_h // 2 + 1)
        smoothed_helix[i] = np.mean(helix_scores[h_start:h_end])

        # Sheet window
        e_start = max(0, i - window_e // 2)
        e_end = min(n, i + window_e // 2 + 1)
        smoothed_sheet[i] = np.mean(sheet_scores[e_start:e_end])

        # Coil window (smaller window)
        c_start = max(0, i - 2)
        c_end = min(n, i + 3)
        smoothed_coil[i] = np.mean(coil_scores[c_start:c_end])

    # Predict structure based on highest propensity
    predictions = []
    confidences = []

    for i in range(n):
        h_score = smoothed_helix[i]
        e_score = smoothed_sheet[i]
        c_score = smoothed_coil[i]

        # Enhanced special residue rules
        aa = sequence[i]

        # Proline is a STRONG helix breaker
        if aa == 'P':
            c_score *= 2.0
            h_score *= 0.2
            e_score *= 0.7

        # Glycine destabilizes regular structures
        if aa == 'G':
            h_score *= 0.4
            e_score *= 0.8
            c_score *= 1.5

        # Charged residues favor coils if isolated
        if aa in ['D', 'E', 'K', 'R']:
            # Check neighbors
            neighbors = sequence[max(0,i-2):min(n,i+3)]
            charged_count = sum(1 for x in neighbors if x in ['D','E','K','R'])
            if charged_count <= 2:  # Isolated charges
                c_score *= 1.3
                h_score *= 0.9

        # Beta-branched residues favor sheets
        if aa in ['V', 'I', 'T']:
            e_score *= 1.2
            h_score *= 0.9

        # Small residues (A, S) in runs favor helices
        if aa in ['A', 'S']:
            neighbors = sequence[max(0,i-3):min(n,i+4)]
            if neighbors.count('A') + neighbors.count('S') >= 3:
                h_score *= 1.1

        # Aromatic clusters favor sheets
        if aa in ['F', 'Y', 'W']:
            neighbors = sequence[max(0,i-2):min(n,i+3)]
            aromatic_count = sum(1 for x in neighbors if x in ['F','Y','W'])
            if aromatic_count >= 2:
                e_score *= 1.3
                h_score *= 0.85

        # N-terminal and C-terminal regions favor coils
        if i < 5 or i >= n - 5:
            c_score *= 1.3
            h_score *= 0.85
            e_score *= 0.85

        # Normalize scores to prevent helix bias
        h_score *= 0.85  # Reduce helix bias
        e_score *= 1.1   # Boost sheet slightly
        c_score *= 1.2   # Boost coil more

        scores = [h_score, e_score, c_score]
        max_score = max(scores)
        total_score = sum(scores)

        # More stringent thresholds
        if c_score == max_score or max_score < 1.0:
            predictions.append(2)  # Coil (default)
        elif h_score == max_score and h_score > 1.08 and h_score > c_score * 1.15:
            predictions.append(0)  # Helix
        elif e_score == max_score and e_score > 1.10 and e_score > c_score * 1.15:
            predictions.append(1)  # Sheet
        else:
            predictions.append(2)  # Coil

        # Calculate confidence
        if total_score > 0:
            confidences.append([h_score/total_score, e_score/total_score, c_score/total_score])
        else:
            confidences.append([0.33, 0.33, 0.34])

    # Post-processing: remove very short helices/sheets (< 3 residues)
    predictions = smooth_predictions(predictions)

    return np.array(predictions), np.array(confidences)

def smooth_predictions(predictions):
    """Remove unrealistically short secondary structure elements"""
    n = len(predictions)
    smoothed = predictions.copy()

    i = 0
    while i < n:
        current = predictions[i]

        # Find run length
        j = i
        while j < n and predictions[j] == current:
            j += 1
        run_length = j - i

        # Helices and sheets should be at least 3-4 residues
        if current in [0, 1] and run_length < 3:
            # Convert short helix/sheet to coil
            for k in range(i, j):
                smoothed[k] = 2

        i = j

    return smoothed

def compute_window_features(sequence, window_size=7):
    """Compute local sequence context features using sliding window"""
    seq_len = len(sequence)
    half_window = window_size // 2

    window_features = []

    for i in range(seq_len):
        start = max(0, i - half_window)
        end = min(seq_len, i + half_window + 1)
        window = sequence[start:end]

        aa_counts = {aa: window.count(aa) / len(window) for aa in AA_LIST}

        hydro_avg = np.mean([AA_PROPERTIES[aa][0] for aa in window if aa in AA_PROPERTIES])
        mw_avg = np.mean([AA_PROPERTIES[aa][1] for aa in window if aa in AA_PROPERTIES])

        h_prop = np.mean([HELIX_PROPENSITY.get(aa, 1.0) for aa in window])
        e_prop = np.mean([SHEET_PROPENSITY.get(aa, 1.0) for aa in window])
        c_prop = np.mean([COIL_PROPENSITY.get(aa, 1.0) for aa in window])

        window_features.append({
            'aa_freq': list(aa_counts.values()),
            'hydro': hydro_avg,
            'mw': mw_avg / 200.0,
            'h_prop': h_prop,
            'e_prop': e_prop,
            'c_prop': c_prop
        })

    return window_features

def sequence_to_features(sequence):
    """Convert amino acid sequence to feature vector with enhanced features"""
    sequence = sequence.upper().strip()
    sequence = re.sub(r'[^ARNDCQEGHILKMFPSTWYV]', '', sequence)

    if len(sequence) == 0:
        return None, "Invalid sequence! Please enter valid amino acids."

    window_features = compute_window_features(sequence)

    features = []
    for idx, aa in enumerate(sequence):
        if aa in AA_LIST:
            onehot = [0.0] * 20
            onehot[AA_LIST.index(aa)] = 1.0

            pos_sin = np.sin(2 * np.pi * idx / len(sequence))
            pos_cos = np.cos(2 * np.pi * idx / len(sequence))
            position_features = [pos_sin, pos_cos]

            window_feat = (
                window_features[idx]['aa_freq'][:20] +
                [window_features[idx]['h_prop'],
                 window_features[idx]['e_prop'],
                 window_features[idx]['c_prop'],
                 window_features[idx]['hydro'],
                 window_features[idx]['mw']] +
                [0.0] * 10
            )

            physchem = AA_PROPERTIES[aa]
            physchem_norm = [
                physchem[0] / 5.0,
                physchem[1] / 200.0,
                physchem[2] / 10.0,
                physchem[3],
                physchem[4] / 15.0,
                physchem[5] / 200.0
            ]

            features.append(onehot + position_features + window_feat[:35] + physchem_norm)

    return np.array(features, dtype=np.float32), None

# ==================== Load Model ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRU_SS_Model().to(device)

try:
    model.load_state_dict(torch.load('best_gru_fusion_cb513.pth', map_location=device))
    model.eval()
    model_loaded = True
    print("✅ Model weights loaded successfully!")
except:
    print("⚠️ Model weights not found. Using Chou-Fasman algorithm for prediction.")
    model_loaded = False

# ==================== Prediction Function ====================
def predict_structure(sequence, show_confidence=True):
    """Predict secondary structure from amino acid sequence"""

    sequence_clean = re.sub(r'[^ARNDCQEGHILKMFPSTWYV]', '', sequence.upper().strip())

    if len(sequence_clean) == 0:
        return "Invalid sequence! Please enter valid amino acids.", None, None

    if len(sequence_clean) > 700:
        return "⚠️ Sequence too long! Maximum length is 700 residues.", None, None

    if len(sequence_clean) < 10:
        return "⚠️ Sequence too short! Minimum length is 10 residues.", None, None

    # Use trained model if available, otherwise use Chou-Fasman
    if model_loaded:
        features, error = sequence_to_features(sequence_clean)
        if error:
            return error, None, None

        if len(features) < 700:
            padding = np.zeros((700 - len(features), 63), dtype=np.float32)
            features = np.vstack([features, padding])

        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            pred_seq = preds[0, :len(sequence_clean)].cpu().numpy()
            prob_seq = probs[0, :len(sequence_clean)].cpu().numpy()

        method = "Deep Learning (GRU)"
    else:
        # Use Chou-Fasman algorithm
        pred_seq, prob_seq = chou_fasman_predict(sequence_clean)
        method = "Chou-Fasman Algorithm"

    # Convert to structure labels
    structure_map = {0: 'H', 1: 'E', 2: 'C'}
    predicted_structure = ''.join([structure_map[p] for p in pred_seq])

    # Calculate statistics
    h_count = predicted_structure.count('H')
    e_count = predicted_structure.count('E')
    c_count = predicted_structure.count('C')
    total = len(predicted_structure)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})

    # Color map for structures
    colors = {'H': '#FF6B6B', 'E': '#4ECDC4', 'C': '#FFE66D'}
    color_list = [colors[s] for s in predicted_structure]

    # Plot 1: Structure prediction
    ax1.bar(range(len(predicted_structure)), [1]*len(predicted_structure),
            color=color_list, width=1.0, edgecolor='none')
    ax1.set_xlim(-0.5, len(predicted_structure)-0.5)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Residue Position', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Structure', fontsize=12, fontweight='bold')
    ax1.set_title(f'🧬 Predicted Secondary Structure ({method})', fontsize=14, fontweight='bold', pad=20)
    ax1.set_yticks([0.5])
    ax1.set_yticklabels(['Structure'])
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label=f'Helix (H): {h_count} ({h_count/total*100:.1f}%)'),
        Patch(facecolor='#4ECDC4', label=f'Strand (E): {e_count} ({e_count/total*100:.1f}%)'),
        Patch(facecolor='#FFE66D', label=f'Coil (C): {c_count} ({c_count/total*100:.1f}%)')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    # Plot 2: Confidence scores
    if show_confidence:
        max_probs = prob_seq.max(axis=1)
        ax2.plot(range(len(max_probs)), max_probs, color='#00D4FF', linewidth=2)
        ax2.fill_between(range(len(max_probs)), max_probs, alpha=0.3, color='#00D4FF')
        ax2.set_xlim(-0.5, len(predicted_structure)-0.5)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Residue Position', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Confidence', fontsize=12, fontweight='bold')
        ax2.set_title('📊 Prediction Confidence', fontsize=14, fontweight='bold', pad=20)
        ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='High confidence')
        ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium confidence')
        ax2.legend(loc='lower right')
        ax2.grid(axis='both', alpha=0.3, linestyle='--')

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()

    avg_confidence = prob_seq.max(axis=1).mean()

    result_text = f"""
## 🎯 Prediction Results

**Method:** {method}
**Sequence Length:** {len(sequence_clean)} residues

**Structure Composition:**
- 🔴 **Helix (H):** {h_count} residues ({h_count/total*100:.1f}%)
- 🔵 **Strand (E):** {e_count} residues ({e_count/total*100:.1f}%)
- 🟡 **Coil (C):** {c_count} residues ({c_count/total*100:.1f}%)

**Average Confidence:** {avg_confidence:.2%}

**Input Sequence:**
```
{sequence_clean}
```

**Predicted Structure:**
```
{predicted_structure}
```

{'✅ Using trained GRU model weights (high accuracy expected)' if model_loaded else '⚠️ Using Chou-Fasman algorithm (trained model not loaded). For better accuracy, load model weights.'}
"""

    return result_text, img, predicted_structure

# ==================== Example Sequences ====================
examples = [
    ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"],
    ["GAMGSMNVQAEEQLQEVASQYRKLLKKELQALLQGQGMSEYDRDGLDAASYYAPVR"],
    ["ARNDCEQGHILKMFPSTWYV"],
    ["MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMARKVLCDSSSSELAVDGFKCCSNSCKSQGSCAEGCKCCEGSCGAGDCQCCRTHFYGGCGGNKNNFCSEERQSCPDHWHYCRSSGSCPCSDCRGACNRCCPHCFCG"],
    ["PEAQRAALERRKQALEQERERLQKELEELKAENERKLQAEIEELRAQAAEERRQRQAAEERRRKQ"],
]

# ==================== Gradio Interface ====================
theme = gr.themes.Soft()

with gr.Blocks(theme=theme, title="Protein Structure Predictor") as demo:

    gr.Markdown(
        """
        # 🧬 Protein Secondary Structure Prediction
        ### GRU Deep Learning + Chou-Fasman Algorithm

        Enter an amino acid sequence to predict its secondary structure (Helix, Strand, or Coil).
        - **With trained model**: Uses bidirectional GRU (~75% accuracy)
        - **Without model**: Uses Chou-Fasman algorithm (~60% accuracy)
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            sequence_input = gr.Textbox(
                label="Amino Acid Sequence",
                placeholder="Enter your protein sequence (e.g., MKTAYIAKQRQISFVK...)",
                lines=5,
                max_lines=10
            )

            with gr.Row():
                predict_btn = gr.Button("🔍 Predict Structure", variant="primary", size="lg")
                clear_btn = gr.ClearButton([sequence_input], value="Clear", size="lg")

            confidence_check = gr.Checkbox(label="Show Confidence Scores", value=True)

            gr.Markdown(
                """
                ### 📝 Instructions:
                1. Enter a protein sequence using standard amino acid codes (20 letters)
                2. Sequence length: 10-700 residues
                3. Invalid characters will be automatically removed
                4. Click "Predict Structure" to analyze

                ### 🎨 Structure Legend:
                - 🔴 **H (Helix):** Alpha helices and 3₁₀ helices
                - 🔵 **E (Strand):** Extended beta strands
                - 🟡 **C (Coil):** Random coils and loops

                ### 🔬 Algorithms:
                - **GRU Model:** Deep learning with 63 features (when weights loaded)
                - **Chou-Fasman:** Rule-based using propensity parameters (fallback)
                """
            )

        with gr.Column(scale=3):
            result_text = gr.Markdown(label="Results")
            structure_viz = gr.Image(label="Structure Visualization", type="pil")
            structure_output = gr.Textbox(label="Raw Structure String", lines=3)

    gr.Markdown("### 📚 Example Sequences")
    gr.Examples(
        examples=examples,
        inputs=sequence_input,
        label="Click to load example sequences"
    )

    gr.Markdown(
        """
        ---
        ### ℹ️ About This Predictor

        **Two-Mode Operation:**

        1. **Deep Learning Mode** (when model weights available):
           - Bidirectional GRU with 128 hidden units
           - 63-dimensional feature vectors
           - Expected Q3 accuracy: ~70-78%

        2. **Chou-Fasman Mode** (fallback):
           - Classic algorithm from 1974
           - Based on amino acid propensities
           - Expected Q3 accuracy: ~50-65%

        **Propensity-Based Predictions:**
        - **Helix formers**: E, A, L, M (hydrophobic, small)
        - **Sheet formers**: V, I, F, Y (branched, aromatic)
        - **Coil formers**: G, P, N, D (flexible, charged)

        **Special Rules:**
        - Proline (P) → Strong helix breaker
        - Glycine (G) → Destabilizes helices
        - Hydrophobic clusters → Promote helices

        Built with PyTorch & Gradio | Hybrid Prediction System
        """
    )

    predict_btn.click(
        fn=predict_structure,
        inputs=[sequence_input, confidence_check],
        outputs=[result_text, structure_viz, structure_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
