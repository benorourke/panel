import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv
from scipy.stats import norm
from holoviews import streams

pn.extension(design='material')


# --- 1. Black-Scholes Logic ---
def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0: return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    return (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))

# --- 2. Simulation State ---
class MarketSim:
    def __init__(self):
        self.S = 100.0
        self.t = 0
        self.dt = 1 / 252
        self.data = pd.DataFrame(columns=['Time', 'Stock_Price', 'Option_Price'])

    def step(self, mu, sigma, K, r, T_expiry, shock=0):
        self.S *= (1 + shock)  # Apply shock if any

        # GBM Step
        epsilon = np.random.normal()
        drift = (mu - 0.5 * sigma ** 2) * self.dt
        diffusion = sigma * np.sqrt(self.dt) * epsilon
        self.S *= np.exp(drift + diffusion)

        # Option Pricing (Time decay: T_expiry - current_time)
        time_to_decay = max(0.001, T_expiry - (self.t * self.dt))
        opt_p = black_scholes(self.S, K, time_to_decay, r, sigma)

        new_row = pd.DataFrame({'Time': [self.t], 'Stock_Price': [self.S], 'Option_Price': [opt_p]})
        self.data = pd.concat([self.data, new_row], ignore_index=True).tail(50)  # Keep last 50 pts
        self.t += 1
        return self.data


sim = MarketSim()

# --- 3. UI Components ---
mu_slider = pn.widgets.FloatSlider(name='Annual Return (μ)', start=-0.2, end=0.2, step=0.01, value=0.05)
sigma_slider = pn.widgets.FloatSlider(name='Volatility (σ)', start=0.05, end=1.0, step=0.01, value=0.2)
strike_slider = pn.widgets.IntSlider(name='Strike Price (K)', start=80, end=120, value=100)
expiry_slider = pn.widgets.FloatSlider(name='Time to Expiry (T)', start=0.1, end=2.0, step=0.1, value=1.0)
shock_button = pn.widgets.Button(name='⚠️ Trigger -10% Shock', button_type='danger')


def get_plots():
    df = sim.data
    if df.empty: return hv.Curve([]) * hv.Curve([])

    stock_curve = hv.Curve(df, 'Time', 'Stock_Price', label='Stock Price').opts(color='blue', width=600, height=300)
    option_curve = hv.Curve(df, 'Time', 'Option_Price', label='Option Price (Call)').opts(color='green', width=600,
                                                                                          height=300)

    return (stock_curve + option_curve).cols(1)


# --- 4. Callbacks ---
def update(event=None):
    shock_val = -0.10 if event else 0
    sim.step(mu_slider.value, sigma_slider.value, strike_slider.value, 0.05, expiry_slider.value, shock=shock_val)
    plot_pane.object = get_plots()


shock_button.on_click(update)
cb = pn.state.add_periodic_callback(update, period=200, count=None)  # 200ms updates

plot_pane = pn.pane.HoloViews(get_plots())

# --- 5. Layout ---
dashboard = pn.Column(
    "# Real-Time Option Pricing Simulator",
    pn.Row(
        pn.Column("### Parameters", mu_slider, sigma_slider, strike_slider, expiry_slider, shock_button),
        plot_pane
    )
)

dashboard.servable()