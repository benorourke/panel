from typing import Literal

import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv
from scipy.stats import norm

pn.extension(design='material')


"""
Mu (Annual Return/Drift): This is the expected total return of the underlying asset in the real world. Used by GBM sim
to calculate the next price of the asset.

It includes a risk premium (the extra return investors demand for holding a risky asset).
"""
UNDERLYING_ANNUAL_RETURN = 0.07
RISK_FREE_RATE = 0.05

def black_scholes(S, K, T, r, sigma, option_type: Literal['call', 'put']):
    """
    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (decimal)
        sigma: Volatility (decimal)
        option_type: "call" or "put"
    """
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = (S * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == "put":
        price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price

# --- 2. Simulation State ---
class MarketSim:
    def __init__(self):
        self.S = 100.0
        self.sigma = 0.2  # Initial volatility
        self.t = 0
        self.dt = 1 / 252
        self.data = pd.DataFrame(columns=['Time', 'Underlying_Price', 'Option_Price', 'Volatility'])

    def step(self, mu, sigma_bounds, K, r, T_expiry, shock=0):
        self.S *= (1 + shock)
        
        # 1. Real-time Volatility Simulation (Random Walk within bounds)
        low, high = sigma_bounds
        vol_shock = np.random.normal(0, 0.02) # Small daily vol change
        self.sigma = np.clip(self.sigma + vol_shock, low, high)

        # 2. GBM Step using simulated sigma
        epsilon = np.random.normal()
        drift = (mu - 0.5 * self.sigma ** 2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * epsilon
        self.S *= np.exp(drift + diffusion)

        # 3. Option Pricing with fixed T_expiry as opposed to iteratively decaying T
        # time_to_decay = max(0.001, T_expiry - (self.t * self.dt))
        # opt_p = black_scholes(self.S, K, time_to_decay, r, self.sigma)
        opt_p = black_scholes(self.S, K, T_expiry, r, self.sigma, option_type='put')
        opt_c = black_scholes(self.S, K, T_expiry, r, self.sigma, option_type='call')



        new_row = pd.DataFrame({
            'Time': [self.t], 
            'Underlying_Price': [self.S],
            'Put_Option_Price': [opt_p],
            'Call_Option_Price': [opt_c],
            'Volatility': [self.sigma]
        })
        self.data = pd.concat([self.data, new_row], ignore_index=True).tail(50)
        self.t += 1
        return self.data


sim = MarketSim()

# --- UI Components ---
sigma_range = pn.widgets.RangeSlider(
    name='Volatility Bounds (σ)', start=0.01, end=1.0, value=(0.15, 0.3), step=0.01
)
strike_slider = pn.widgets.IntSlider(name='Strike Price (K)', start=50, end=150, value=100)
expiry_slider = pn.widgets.FloatSlider(name='Time to Expiry (T)', start=0.1, end=2.0, step=0.1, value=1.0)
shock_button = pn.widgets.Button(name='💣 Trigger -10% Shock', button_type='danger')
pause_button = pn.widgets.Toggle(name='Pause Simulation', button_type='primary', value=False)

def get_plots():
    df = sim.data
    if df.empty: return hv.Curve([]) * hv.Curve([])

    # Adjusting width to ~400 so two plots fit side-by-side comfortably
    vol_curve = hv.Curve(df, 'Time', 'Volatility', label='Simulated Vol (σ)').opts(color='red', width=400, height=250)
    underlying_curve = hv.Curve(df, 'Time', 'Underlying_Price', label='Underlying Price').opts(color='blue', width=400, height=250)
    put_option_curve = hv.Curve(df, 'Time', 'Put_Option_Price', label='Option Price (Put)').opts(color='green', width=400, height=250)
    call_option_curve = hv.Curve(df, 'Time', 'Call_Option_Price', label='Option Price (Call)').opts(color='purple', width=400, height=250)

    # Creating the 2x2 grid layout
    return (vol_curve + underlying_curve + put_option_curve + call_option_curve).cols(2)


# --- 4. Callbacks ---
def update(event=None):
    shock_val = -0.10 if event else 0
    sim.step(
        UNDERLYING_ANNUAL_RETURN,
        sigma_range.value,
        strike_slider.value,
        RISK_FREE_RATE,
        expiry_slider.value,
        shock=shock_val
    )
    plot_pane.object = get_plots()


def toggle_pause(event):
    if event.new:
        cb.stop()
        pause_button.name = '▶ Resume Simulation'
    else:
        cb.start()
        pause_button.name = 'Pause Simulation'


pause_button.param.watch(toggle_pause, 'value')
shock_button.on_click(update)
cb = pn.state.add_periodic_callback(update, period=200, count=None)  # 200ms updates

plot_pane = pn.pane.HoloViews(get_plots())

# --- 5. Layout ---
dashboard = pn.Column(
    "# Option Pricing & Volatility Simulator",
    pn.Row(
        pn.Column("### Config", sigma_range, strike_slider, expiry_slider, shock_button, pause_button),
        plot_pane
    )
)

dashboard.servable()