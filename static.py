import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv

pn.extension()
hv.extension("bokeh")

# -----------------------------
# Sample order book data
# -----------------------------
np.random.seed(1)

# Bid side (descending prices)
bid_prices = np.arange(99.5, 95, -0.5)
bid_sizes = np.random.randint(10, 100, len(bid_prices))

# Ask side (ascending prices)
ask_prices = np.arange(100.5, 105, 0.5)
ask_sizes = np.random.randint(10, 100, len(ask_prices))

bids = pd.DataFrame({
    "price": bid_prices,
    "volume": bid_sizes
}).sort_values("price", ascending=False)

asks = pd.DataFrame({
    "price": ask_prices,
    "volume": ask_sizes
}).sort_values("price")

# Cumulative volume
bids["cum_volume"] = bids["volume"].cumsum()
asks["cum_volume"] = asks["volume"].cumsum()

# -----------------------------
# Depth curves
# -----------------------------
bid_curve = hv.Curve(
    bids, kdims="price", vdims="cum_volume", label="Bids"
).opts(
    color="green",
    line_width=2,
    line_dash="solid",
    interpolation="steps-post"
)

ask_curve = hv.Curve(
    asks, kdims="price", vdims="cum_volume", label="Asks"
).opts(
    color="red",
    line_width=2,
    interpolation="steps-pre"
)

# Mid-price marker
mid_price = (bids["price"].iloc[0] + asks["price"].iloc[0]) / 2
mid_line = hv.VLine(mid_price).opts(
    color="gray",
    line_dash="dashed",
    line_width=1
)

# -----------------------------
# Final chart
# -----------------------------
depth_chart = (bid_curve * ask_curve * mid_line).opts(
    height=450,
    width=800,
    xlabel="Price",
    ylabel="Cumulative Volume",
    title="Order Book Depth Chart",
    legend_position="top_left",
    tools=["hover"]
)

# -----------------------------
# Panel layout
# -----------------------------
app = pn.Column(
    "## 📊 Exchange Order Depth",
    depth_chart
)

app.servable()