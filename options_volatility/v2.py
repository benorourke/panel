from typing import Literal

import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv
from scipy.stats import norm

pn.extension(design='material')

"""
Mu (Annual Return/Drift): This is the expected total return of the underlying asset in the real world. 
Used by GBM sim to calculate the next price of the asset.
"""
UNDERLYING_ANNUAL_RETURN = 0.07
RISK_FREE_RATE = 0.05


def black_scholes(S, K, T, r, sigma, option_type: Literal['call', 'put']):
    """
    Standard Black-Scholes pricing model.
    """
    # Convert T from days to years
    T_years = T / 252

    # Avoid division by zero if T is very close to 0
    if T_years <= 1e-5:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T_years) / (sigma * np.sqrt(T_years))
    d2 = d1 - sigma * np.sqrt(T_years)

    if option_type == "call":
        price = (S * norm.cdf(d1)) - (K * np.exp(-r * T_years) * norm.cdf(d2))
    elif option_type == "put":
        price = (K * np.exp(-r * T_years) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price


class PricingSimulator:
    def __init__(self):
        self.S = 100.0
        self.sigma = 0.2  # Initial volatility
        self.t = 0
        self.dt = 1 / 252
        self.data = pd.DataFrame(columns=['Time', 'Underlying_Price', 'Option_Price', 'Volatility'])

    def step(self, mu, sigma_bounds, K, r, T_expiry, shock=0):
        # Apply Market Shock
        self.S *= (1 + shock)

        # 1. Real-time Volatility Simulation (Random Walk within bounds)
        low, high = sigma_bounds
        vol_shock = np.random.normal(0, 0.02)  # Small daily vol change
        self.sigma = np.clip(self.sigma + vol_shock, low, high)

        # 2. GBM Step using simulated sigma
        epsilon = np.random.normal()
        drift = (mu - 0.5 * self.sigma ** 2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * epsilon
        self.S *= np.exp(drift + diffusion)

        # 3. Option Pricing
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


sim = PricingSimulator()

# --- UI Components ---

# Group 1: Market Params
sigma_range = pn.widgets.RangeSlider(
    name='Volatility Bounds (σ)', start=0.01, end=1.0, value=(0.15, 0.3), step=0.01
)
strike_slider = pn.widgets.IntSlider(name='Strike Price (K)', start=50, end=150, value=100)
expiry_slider = pn.widgets.IntSlider(name='Time to Expiry (Days)', start=1, end=252, step=1, value=30)

# Group 2: Simulation Controls
shock_input = pn.widgets.FloatInput(
    name='Shock Size (%)', value=-10.0, step=1.0, start=-50.0, end=50.0, width=100
)
shock_button = pn.widgets.Button(
    name='💥 Apply Shock', button_type='danger', width=130
)
pause_button = pn.widgets.Toggle(
    name='⏸️ Pause', button_type='primary', value=False, width=240
)


def get_plots():
    df = sim.data
    if df.empty: return hv.Curve([]) * hv.Curve([])

    # Plot styling
    common_opts = dict(height=250, responsive=True, gridstyle={'grid_line_color': '#efefef'})

    vol_curve = hv.Curve(df, 'Time', 'Volatility', label='Simulated Vol (σ)').opts(
        color='#FF5722', title="Volatility Process", **common_opts
    )
    underlying_curve = hv.Curve(df, 'Time', 'Underlying_Price', label='Price').opts(
        color='#2196F3', title="Underlying Asset", **common_opts
    )
    put_curve = hv.Curve(df, 'Time', 'Put_Option_Price', label='Put').opts(
        color='#4CAF50', title="Put Option", **common_opts
    )
    call_curve = hv.Curve(df, 'Time', 'Call_Option_Price', label='Call').opts(
        color='#9C27B0', title="Call Option", **common_opts
    )

    # 2x2 Grid
    return (vol_curve + underlying_curve + put_curve + call_curve).cols(2)


# --- Callbacks ---
def update(event=None):
    # Determine shock value
    # We check if the event object matches the shock_button to ensure
    # the periodic callback (event=None) doesn't trigger a shock.
    shock_val = 0.0
    if event is not None and event.obj == shock_button:
        shock_val = shock_input.value / 100.0

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
        pause_button.name = '▶ Resume'
        pause_button.button_type = 'success'
    else:
        cb.start()
        pause_button.name = '⏸️ Pause'
        pause_button.button_type = 'primary'


pause_button.param.watch(toggle_pause, 'value')
shock_button.on_click(update)
cb = pn.state.add_periodic_callback(update, period=200, count=None)

plot_pane = pn.pane.HoloViews(get_plots(), sizing_mode='stretch_width')

# --- Layout ---

# Create formatted Cards for the sidebar
market_card = pn.Card(
    sigma_range, strike_slider, expiry_slider,
    title="📉 Market Parameters",
    header_background='#f0f0f0',
    collapsed=False
)

sim_card = pn.Card(
    pn.Row(shock_input, shock_button, align='end'),  # Align input and button
    pn.layout.Divider(),
    pause_button,
    title="⚡ Simulation Controls",
    header_background='#f0f0f0',
    collapsed=False
)

# Main Template
dashboard = pn.template.MaterialTemplate(
    title="Option Pricing & Volatility Simulator",
    sidebar=[market_card, sim_card],
    main=[plot_pane],
)

dashboard.servable()