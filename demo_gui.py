"""
demo_gui.py
-----------
Tkinter-based visual demo application.
Classifies emotions from a selected WAV file.

Usage:
    python demo_gui.py

Requirements:
    pip install librosa scikit-learn joblib numpy soundfile
"""

import os
import threading
import warnings
warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

_predict_func = None

EMOTION_EMOJI = {
    "happy"    : "😊",
    "sad"      : "😢",
    "angry"    : "😠",
    "furious"  : "🤬",
    "neutral"  : "😐",
    "surprised": "😲",
}

EMOTION_COLORS = {
    "happy"    : "#F5C518",
    "sad"      : "#4A90D9",
    "angry"    : "#E74C3C",
    "furious"  : "#C0392B",
    "neutral"  : "#95A5A6",
    "surprised": "#9B59B6",
}

BG       = "#1A1A2E"
CARD_BG  = "#16213E"
ACCENT   = "#0F3460"
FG       = "#E0E0E0"
FG_DIM   = "#888888"
FONT_BIG = ("Segoe UI", 28, "bold")
FONT_MED = ("Segoe UI", 13)
FONT_SM  = ("Segoe UI", 10)


def load_predictor():
    global _predict_func
    if _predict_func is None:
        from predict import predict_file
        _predict_func = predict_file
    return _predict_func


class EmotionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🎙️  Emotion Classifier  |  COE216 Final")
        self.geometry("720x600")
        self.resizable(False, False)
        self.configure(bg=BG)

        self._build_ui()
        self._check_model()

    # -- UI construction -----------------------------------------------------

    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=30, pady=(25, 10))
        tk.Label(header, text="🎙️  Emotion Classifier",
                 bg=BG, fg=FG, font=("Segoe UI", 20, "bold")).pack(side="left")
        tk.Label(header, text="COE216 Final  •  Emo Challenge 2026",
                 bg=BG, fg=FG_DIM, font=FONT_SM).pack(side="right", anchor="s")

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=30, pady=5)

        # File selection card
        file_card = tk.Frame(self, bg=CARD_BG, bd=0, relief="flat")
        file_card.pack(fill="x", padx=30, pady=10)
        file_card.pack_propagate(False)
        file_card.configure(height=90)

        tk.Label(file_card, text="WAV File", bg=CARD_BG, fg=FG_DIM,
                 font=FONT_SM).place(x=15, y=10)

        self.file_label = tk.Label(file_card, text="No file selected",
                                   bg=CARD_BG, fg=FG, font=FONT_SM,
                                   wraplength=480, justify="left")
        self.file_label.place(x=15, y=32)

        btn_browse = tk.Button(file_card, text="Browse",
                               bg=ACCENT, fg=FG, font=FONT_SM,
                               relief="flat", cursor="hand2", padx=12, pady=4,
                               command=self._browse_file)
        btn_browse.place(x=620, y=30)

        # Analyze button
        self.btn_analyze = tk.Button(
            self, text="▶  Analyze", bg="#E94560", fg="white",
            font=("Segoe UI", 14, "bold"), relief="flat", cursor="hand2",
            padx=20, pady=10, command=self._run_analysis
        )
        self.btn_analyze.pack(pady=10)

        # Result panel
        result_frame = tk.Frame(self, bg=CARD_BG)
        result_frame.pack(fill="both", padx=30, pady=(5, 15), expand=True)

        self.emotion_label = tk.Label(result_frame, text="—", bg=CARD_BG, fg=FG,
                                      font=("Segoe UI", 52, "bold"))
        self.emotion_label.pack(pady=(20, 5))

        self.emotion_name = tk.Label(result_frame, text="Awaiting prediction...",
                                     bg=CARD_BG, fg=FG_DIM, font=FONT_MED)
        self.emotion_name.pack()

        # Probability bars
        self.bar_frame = tk.Frame(result_frame, bg=CARD_BG)
        self.bar_frame.pack(fill="x", padx=25, pady=15)

        # Status bar
        self.status_var = tk.StringVar(value="Loading model...")
        tk.Label(self, textvariable=self.status_var, bg=BG, fg=FG_DIM,
                 font=FONT_SM).pack(side="bottom", pady=5)

        self.selected_file = None

    def _check_model(self):
        try:
            load_predictor()
            if not all(os.path.exists(p) for p in
                       ["model.joblib", "scaler.joblib", "label_encoder.joblib"]):
                self.status_var.set("⚠️  Model not found — run train.py first")
            else:
                self.status_var.set("✅  Model ready")
        except Exception as e:
            self.status_var.set(f"⚠️  {e}")

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select WAV File",
            filetypes=[("WAV Files", "*.wav"), ("All Files", "*.*")]
        )
        if path:
            self.selected_file = path
            name = os.path.basename(path)
            self.file_label.config(text=f"📁  {name}")
            self.status_var.set(f"File selected: {name}")

    def _run_analysis(self):
        if not self.selected_file:
            messagebox.showwarning("No File", "Please select a WAV file first.")
            return

        self.btn_analyze.config(state="disabled", text="⏳  Analyzing...")
        self.status_var.set("Extracting features...")
        self.emotion_label.config(text="⏳")
        self.emotion_name.config(text="Analyzing, please wait...")

        thread = threading.Thread(target=self._analyze_thread, daemon=True)
        thread.start()

    def _analyze_thread(self):
        try:
            predict_fn = load_predictor()
            label, conf = predict_fn(self.selected_file)
            self.after(0, lambda: self._show_result(label, conf))
        except Exception as e:
            self.after(0, lambda: self._show_error(str(e)))

    def _show_result(self, label: str, conf: dict):
        emoji = EMOTION_EMOJI.get(label, "🎵")
        color = EMOTION_COLORS.get(label, "#FFFFFF")

        self.emotion_label.config(text=emoji, fg=color)
        self.emotion_name.config(
            text=label.upper(), fg=color, font=("Segoe UI", 18, "bold")
        )

        # Clear and redraw probability bars
        for w in self.bar_frame.winfo_children():
            w.destroy()

        if conf:
            sorted_conf = sorted(conf.items(), key=lambda x: -x[1])
            for emotion, prob in sorted_conf:
                row = tk.Frame(self.bar_frame, bg=CARD_BG)
                row.pack(fill="x", pady=2)

                tk.Label(row, text=f"{EMOTION_EMOJI.get(emotion, '')} {emotion:<12}",
                         bg=CARD_BG, fg=FG, font=FONT_SM, width=16, anchor="w"
                         ).pack(side="left")

                canvas = tk.Canvas(row, height=18, bg="#2A2A3E",
                                   highlightthickness=0, bd=0)
                canvas.pack(side="left", fill="x", expand=True, padx=5)
                canvas.update_idletasks()
                w = canvas.winfo_width() or 300
                bar_w = int(w * prob)
                bar_color = EMOTION_COLORS.get(emotion, "#555")
                if bar_w > 0:
                    canvas.create_rectangle(0, 2, bar_w, 16, fill=bar_color, outline="")

                tk.Label(row, text=f"{prob:.1%}", bg=CARD_BG, fg=FG_DIM,
                         font=FONT_SM, width=7).pack(side="right")

        self.btn_analyze.config(state="normal", text="▶  Analyze")
        self.status_var.set(f"✅  Prediction complete: {label.upper()}")

    def _show_error(self, msg: str):
        self.emotion_label.config(text="❌")
        self.emotion_name.config(text="An error occurred", fg="#E74C3C")
        self.btn_analyze.config(state="normal", text="▶  Analyze")
        self.status_var.set(f"Error: {msg}")
        messagebox.showerror("Error", msg)


if __name__ == "__main__":
    app = EmotionApp()
    app.mainloop()
