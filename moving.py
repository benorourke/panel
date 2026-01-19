import numpy as np
import pandas as pd
import panel as pn
import holoviews as hv
from holoviews.streams import Stream

pn.extension()
hv.extension("bokeh")


# -----------------------------
# Data Generation Function
# -----------------------------
def get_order_book_data():
    # Simulate a fluctuating mid-price
    base_price = 100 + np.random.uniform(-0.5, 0.5)

    # Bid side
    bid_prices = np.sort(base_price - np.random.uniform(0.1, 5, 10))[::-1]
    bid_sizes = np.random.randint(10, 100, len(bid_prices))

    # Ask side
    ask_prices = np.sort(base_price + np.random.uniform(0.1, 5, 10))
    ask_sizes = np.random.randint(10, 100, len(ask_prices))

    bids = pd.DataFrame({"price": bid_prices, "volume": bid_sizes})
    asks = pd.DataFrame({"price": ask_prices, "volume": ask_sizes})

    bids["cum_volume"] = bids["volume"].cumsum()
    asks["cum_volume"] = asks["volume"].cumsum()

    return bids, asks


# -----------------------------
# Dynamic Plotting Function
# -----------------------------
def create_depth_chart():
    bids, asks = get_order_book_data()

    bid_curve = hv.Curve(bids, "price", "cum_volume", label="Bids").opts(
        color="green", line_width=2, interpolation="steps-post"
    )

    ask_curve = hv.Curve(asks, "price", "cum_volume", label="Asks").opts(
        color="red", line_width=2, interpolation="steps-pre"
    )

    mid_price = (bids["price"].iloc[0] + asks["price"].iloc[0]) / 2
    mid_line = hv.VLine(mid_price).opts(color="gray", line_dash="dashed")

    return (bid_curve * ask_curve * mid_line).opts(
        height=450, width=800, xlabel="Price", ylabel="Cumulative Volume",
        title="Live Order Book Depth", tools=["hover"]
    )

stream = Stream.define('Next')()
depth_chart = hv.DynamicMap(create_depth_chart, streams=[stream])

# -----------------------------
# Panel Layout & Callback
# -----------------------------
app = pn.Column(
    "## 📊 Live Exchange Order Depth",
    depth_chart
)

# Add a periodic callback to trigger updates every 500ms
pn.state.add_periodic_callback(lambda: depth_chart.event(), period=500)

app.servable()