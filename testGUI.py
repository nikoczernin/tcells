import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from RL.utils import rational_function

# =========================
# Placeholder function c(t, lr, p, bias)
# Replace this with your real implementation. It must return values in [0, 1].
# t is a numpy array.
# =========================
def c(t, lr, p, bias):
    bias = bias / (1 - bias + (1e-12 if bias == 1 else 0))
    t = np.asarray(t, dtype=float)
    c_pos = rational_function(t * bias, flatness=lr)
    c_neg = 1 - rational_function(t / bias, flatness=lr)
    # get the expected value
    certainty = p * c_pos + (1 - p) * c_neg
    return certainty


class CurveExplorerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Curve Explorer: c(t, lr, p, bias)")
        self.geometry("1000x650")

        # ---------- State vars ----------
        # lr uses a log slider: value = 10 ** lr_log10
        self.lr_log10_var = tk.DoubleVar(value=0.0)   # lr = 1 initially
        self.p_var        = tk.DoubleVar(value=0.5)
        self.bias_var     = tk.DoubleVar(value=0.5)
        self.maxT_var     = tk.IntVar(value=100)
        self.logx_var     = tk.BooleanVar(value=False)

        # ---------- Layout ----------
        self._build_ui()

        # Initial plot
        self.update_plot()

    def _build_ui(self):
        # Left panel: controls
        ctrl = ttk.Frame(self, padding=12)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)

        # Right: figure
        fig_frame = ttk.Frame(self)
        fig_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(6, 4.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("c(t, lr, p, bias)")
        self.ax.set_xlabel("t")
        self.ax.set_ylabel("c(t)")
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Optional toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, fig_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # ---------- Controls ----------
        # lr (log slider)
        ttk.Label(ctrl, text="lr (>0, log-scaled)").grid(row=0, column=0, sticky="w")
        self.lr_val_lbl = ttk.Label(ctrl, text="lr = 1.000")
        self.lr_val_lbl.grid(row=0, column=1, sticky="e")

        # Slider for log10(lr), symmetric resolution around lr=1 (log10=0)
        lr_scale = ttk.Scale(
            ctrl, from_=-3.0, to=3.0, orient="horizontal",
            command=lambda _: self._on_slider_change()
        )
        lr_scale.set(self.lr_log10_var.get())
        lr_scale.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 10))
        # tie the widget to the var
        def on_lr_move(val):
            try:
                self.lr_log10_var.set(float(val))
            finally:
                self._on_slider_change()
        lr_scale.configure(command=on_lr_move)

        # p in (0,1)
        ttk.Label(ctrl, text="p (0 < p < 1)").grid(row=2, column=0, sticky="w")
        self.p_val_lbl = ttk.Label(ctrl, text="p = 0.500")
        self.p_val_lbl.grid(row=2, column=1, sticky="e")

        p_scale = ttk.Scale(
            ctrl, from_=0.0, to=1.0, orient="horizontal",
            command=lambda v: self._on_var_slide(self.p_var, v)
        )
        p_scale.set(self.p_var.get())
        p_scale.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(2, 10))

        # bias in (0,1)
        ttk.Label(ctrl, text="bias (0 < bias < 1)").grid(row=4, column=0, sticky="w")
        self.bias_val_lbl = ttk.Label(ctrl, text="bias = 0.500")
        self.bias_val_lbl.grid(row=4, column=1, sticky="e")

        bias_scale = ttk.Scale(
            ctrl, from_=0.0, to=1.0, orient="horizontal",
            command=lambda v: self._on_var_slide(self.bias_var, v)
        )
        bias_scale.set(self.bias_var.get())
        bias_scale.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(2, 10))

        # maxT (integer, time horizon)
        ttk.Label(ctrl, text="maxT").grid(row=6, column=0, sticky="w")
        self.maxT_val_lbl = ttk.Label(ctrl, text="maxT = 100")
        self.maxT_val_lbl.grid(row=6, column=1, sticky="e")

        maxT_scale = ttk.Scale(
            ctrl, from_=10, to=5000, orient="horizontal",
            command=lambda v: self._on_maxT_slide(v)
        )
        maxT_scale.set(self.maxT_var.get())
        maxT_scale.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(2, 10))

        # Log-scale checkbox
        log_chk = ttk.Checkbutton(
            ctrl, text="Log-scale x-axis",
            variable=self.logx_var,
            command=self.update_plot
        )
        log_chk.grid(row=8, column=0, columnspan=2, sticky="w", pady=(4, 12))

        # Hint for log x: t=0 cannot be shown
        ttk.Label(
            ctrl,
            text="Note: with log x-axis, t=0 is omitted (starts near 0).",
            foreground="#666"
        ).grid(row=9, column=0, columnspan=2, sticky="w", pady=(0, 8))

        # Tighten grid
        ctrl.columnconfigure(0, weight=1)
        ctrl.columnconfigure(1, weight=0)

    # ----- Callbacks -----
    def _on_slider_change(self):
        # Update numeric labels and plot
        lr = 10 ** self.lr_log10_var.get()
        self.lr_val_lbl.config(text=f"lr = {lr:.3f}")
        self.update_plot()

    def _on_var_slide(self, var, value):
        var.set(float(value))
        if var is self.p_var:
            self.p_val_lbl.config(text=f"p = {self.p_var.get():.3f}")
        elif var is self.bias_var:
            self.bias_val_lbl.config(text=f"bias = {self.bias_var.get():.3f}")
        self.update_plot()

    def _on_maxT_slide(self, value):
        # Round to int for display; keep slider as float for smoothness
        v = int(float(value))
        self.maxT_var.set(v)
        self.maxT_val_lbl.config(text=f"maxT = {v}")
        self.update_plot()

    # ----- Plotting -----
    def update_plot(self):
        # Read current controls
        lr   = 10 ** self.lr_log10_var.get()
        p    = float(self.p_var.get())
        bias = float(self.bias_var.get())
        maxT = int(self.maxT_var.get())
        logx = bool(self.logx_var.get())

        # Generate t
        n_points = 600
        if logx:
            # log-scale x cannot include 0; start near 0
            eps = max(1e-6, maxT / 1e9)
            t = np.geomspace(eps, maxT, num=n_points)
        else:
            t = np.linspace(0.0, maxT, num=n_points)

        # Compute y
        y = c(t, lr, p, bias)

        # Draw
        self.ax.clear()
        self.ax.plot(t, y, linewidth=2.0)
        self.ax.set_ylabel("c(t)")
        self.ax.set_ylim(0.0, 1.0)
        self.ax.grid(True, which="both", alpha=0.3)

        if logx:
            self.ax.set_xscale("log")
            self.ax.set_xlim(left=max(t.min(), 1e-12), right=maxT)
        else:
            self.ax.set_xscale("linear")
            self.ax.set_xlim(0.0, float(maxT))

        self.ax.set_title("c(t, lr, p, bias)")
        self.ax.set_xlabel("t (log scale)" if logx else "t")
        self.canvas.draw_idle()


if __name__ == "__main__":
    app = CurveExplorerApp()
    app.mainloop()
