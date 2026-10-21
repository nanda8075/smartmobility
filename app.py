from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Smart Mobility Command Center",
    page_icon=":bus:",
    layout="wide",
    initial_sidebar_state="expanded",
)


ROOT = Path(__file__).resolve().parent
BUS_PATH = ROOT / "bigdata" / "datasets" / "bus_data.csv"
PARKING_PATH = ROOT / "bigdata" / "datasets" / "parking_data.csv"
BUS_SPARK_PATH = ROOT / "bigdata" / "output" / "bus_analytics" / "bus_analytics.csv"
PARKING_SPARK_PATH = ROOT / "bigdata" / "output" / "parking_analytics" / "parking_analytics.csv"

BUS_COLUMNS = {
    "bus_id",
    "bus_type",
    "operator",
    "route",
    "from_stop",
    "to_stop",
    "scheduled_time",
    "actual_time",
}
PARKING_COLUMNS = {
    "parking_id",
    "location",
    "type",
    "capacity",
    "occupied",
    "timestamp",
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { --navy: #071522; --navy-2: #0d2233; --teal: #16c6b7; --muted: #8ca7b7; }
        .stApp { background: #071522; color: #eaf4f6; }
        [data-testid="stSidebar"] { background: #0a1d2b; border-right: 1px solid #163747; }
        [data-testid="stSidebar"] .stRadio label { color: #bad0d7; }
        [data-testid="stMetric"] { background: #0d2535; border: 1px solid #174356; border-radius: 12px; padding: 16px; }
        [data-testid="stMetricLabel"] { color: #8ca7b7; }
        [data-testid="stMetricValue"] { color: #efffff; }
        .hero { border-bottom: 1px solid #174356; padding: 8px 0 24px; margin-bottom: 24px; }
        .eyebrow { color: #16c6b7; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; }
        .hero h1 { color: #efffff; font-size: 2.4rem; margin: 5px 0; }
        .hero p, .caption { color: #8ca7b7; }
        .section-title { color: #efffff; font-size: 1.22rem; font-weight: 700; margin: 22px 0 10px; }
        .status { color: #72e3d7; background: #103d42; border-radius: 999px; padding: 5px 10px; font-size: .78rem; font-weight: 700; display: inline-block; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data(path: Path, required_columns: set[str]) -> tuple[pd.DataFrame | None, str | None]:
    if not path.exists():
        return None, f"Missing file: {path}"
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        return None, f"Could not read {path.name}: {error}"
    missing = required_columns.difference(frame.columns)
    if missing:
        return None, f"{path.name} is missing columns: {', '.join(sorted(missing))}"
    return frame, None


def format_number(value: float) -> str:
    return f"{value:,.0f}"


def metric_cards(bus: pd.DataFrame, parking: pd.DataFrame) -> None:
    total_capacity = parking["capacity"].sum()
    total_occupied = parking["occupied"].sum()
    occupancy = total_occupied / total_capacity * 100 if total_capacity else 0
    cards = st.columns(4)
    cards[0].metric("BUS SERVICES", format_number(len(bus)), "scheduled records")
    cards[1].metric("AVG DELAY", f"{bus['delay_minutes'].mean():.1f} min", "from scheduled time")
    cards[2].metric("PARKING OCCUPANCY", f"{occupancy:.1f}%", f"{format_number(total_occupied)} occupied")
    cards[3].metric("AVAILABLE SPACES", format_number(parking["available"].sum()), "across all locations")


def prepare_data(bus: pd.DataFrame, parking: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bus = bus.copy()
    scheduled = pd.to_datetime(bus["scheduled_time"], format="%H:%M", errors="coerce")
    actual = pd.to_datetime(bus["actual_time"], format="%H:%M", errors="coerce")
    bus["delay_minutes"] = (actual - scheduled).dt.total_seconds() / 60
    bus["delay_minutes"] = bus["delay_minutes"].fillna(0).clip(lower=0)
    parking = parking.copy()
    for column in ["capacity", "occupied"]:
        parking[column] = pd.to_numeric(parking[column], errors="coerce").fillna(0)
    parking["available"] = (parking["capacity"] - parking["occupied"]).clip(lower=0)
    parking["occupancy_pct"] = parking["occupied"].div(parking["capacity"].replace(0, pd.NA)).fillna(0) * 100
    parking["timestamp"] = pd.to_datetime(parking["timestamp"], errors="coerce")
    return bus, parking


def bus_analytics(bus: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Bus delay performance</div>', unsafe_allow_html=True)
    left, right = st.columns([1.35, 1])
    route_delay = bus.groupby("route", as_index=False)["delay_minutes"].mean().sort_values("delay_minutes", ascending=False)
    with left:
        chart = px.bar(route_delay, x="route", y="delay_minutes", color="delay_minutes", color_continuous_scale=["#16c6b7", "#f5a65b"], labels={"delay_minutes": "Average delay (minutes)", "route": "Route"})
        chart.update_layout(height=380, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, margin=dict(l=0, r=0, t=25, b=0))
        st.plotly_chart(chart, use_container_width=True)
    with right:
        operator_delay = bus.groupby("operator", as_index=False)["delay_minutes"].mean().sort_values("delay_minutes", ascending=False)
        chart = px.bar(operator_delay, x="delay_minutes", y="operator", orientation="h", color="delay_minutes", color_continuous_scale=["#16c6b7", "#f5a65b"], labels={"delay_minutes": "Average delay (minutes)", "operator": "Operator"})
        chart.update_layout(height=380, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, margin=dict(l=0, r=0, t=25, b=0))
        st.plotly_chart(chart, use_container_width=True)
    st.markdown('<div class="section-title">Service records</div>', unsafe_allow_html=True)
    route_search = st.text_input("Search routes", placeholder="Type a route name...", key="bus_search")
    filtered = bus[bus["route"].str.contains(route_search, case=False, na=False)] if route_search else bus
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button("Download bus data", filtered.to_csv(index=False), "bus_analytics.csv", "text/csv")


def parking_analytics(parking: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Parking network utilization</div>', unsafe_allow_html=True)
    locations = sorted(parking["location"].dropna().unique())
    selected = st.multiselect("Filter parking locations", locations, default=locations, key="parking_locations")
    filtered = parking[parking["location"].isin(selected)]
    left, right = st.columns(2)
    by_location = filtered.groupby("location", as_index=False).agg(capacity=("capacity", "sum"), occupied=("occupied", "sum"), available=("available", "sum"))
    with left:
        occupancy_chart = px.bar(by_location, x="location", y=["occupied", "available"], barmode="stack", color_discrete_sequence=["#16c6b7", "#2d5364"], labels={"value": "Spaces", "variable": "Status", "location": "Location"})
        occupancy_chart.update_layout(height=380, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=25, b=0), legend_title_text="")
        st.plotly_chart(occupancy_chart, use_container_width=True)
    with right:
        type_summary = filtered.groupby("type", as_index=False).agg(occupied=("occupied", "sum"), available=("available", "sum"))
        type_chart = px.bar(type_summary, x="type", y=["occupied", "available"], barmode="group", color_discrete_sequence=["#f5a65b", "#16c6b7"], labels={"value": "Spaces", "variable": "Status", "type": "Parking type"})
        type_chart.update_layout(height=380, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=25, b=0), legend_title_text="")
        st.plotly_chart(type_chart, use_container_width=True)
    st.markdown('<div class="section-title">Parking inventory</div>', unsafe_allow_html=True)
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button("Download parking data", filtered.to_csv(index=False), "parking_analytics.csv", "text/csv")


def overview(bus: pd.DataFrame, parking: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Network pulse</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown("**Routes by average delay**")
        route_delay = bus.groupby("route", as_index=False)["delay_minutes"].mean().sort_values("delay_minutes", ascending=False)
        st.dataframe(route_delay.rename(columns={"route": "Route", "delay_minutes": "Avg delay (min)"}), use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Location occupancy snapshot**")
        snapshot = parking.groupby("location", as_index=False).agg(occupied=("occupied", "sum"), capacity=("capacity", "sum"))
        snapshot["occupancy"] = snapshot["occupied"].div(snapshot["capacity"].replace(0, pd.NA)).fillna(0) * 100
        st.dataframe(snapshot.rename(columns={"location": "Location", "occupancy": "Occupancy (%)"})[["Location", "Occupancy (%)"]], use_container_width=True, hide_index=True)


def reports(bus: pd.DataFrame, parking: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Export centre</div>', unsafe_allow_html=True)
    st.write("Download the processed datasets, including calculated delay, availability, and occupancy fields.")
    bus_report = bus.to_csv(index=False).encode("utf-8")
    parking_report = parking.to_csv(index=False).encode("utf-8")
    first, second = st.columns(2)
    with first:
        st.download_button("Download full bus report", bus_report, "smart_mobility_bus_report.csv", "text/csv", use_container_width=True)
    with second:
        st.download_button("Download full parking report", parking_report, "smart_mobility_parking_report.csv", "text/csv", use_container_width=True)
    st.markdown('<div class="section-title">Data quality</div>', unsafe_allow_html=True)
    quality = pd.DataFrame({"Dataset": ["Bus services", "Parking inventory"], "Rows": [len(bus), len(parking)], "Source": [BUS_PATH.name, PARKING_PATH.name], "Status": ["Loaded", "Loaded"]})
    st.dataframe(quality, use_container_width=True, hide_index=True)


def main() -> None:
    inject_styles()
    bus, bus_error = load_data(BUS_PATH, BUS_COLUMNS)
    parking, parking_error = load_data(PARKING_PATH, PARKING_COLUMNS)
    if bus_error or parking_error:
        st.error("The dashboard could not load the required data files.")
        if bus_error:
            st.warning(bus_error)
        if parking_error:
            st.warning(parking_error)
        st.info("Place bus_data.csv and parking_data.csv in bigdata/datasets, then reload the app.")
        return
    bus, parking = prepare_data(bus, parking)
    st.sidebar.markdown("## SMART MOBILITY")
    st.sidebar.caption("Operations command center")
    page = st.sidebar.radio("Navigate", ["Overview", "Bus Analytics", "Parking Analytics", "Reports"], label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.markdown('<span class="status">● DATA CONNECTED</span>', unsafe_allow_html=True)
    st.sidebar.caption(f"Bus source: {BUS_PATH.name}\n\nParking source: {PARKING_PATH.name}")
    st.markdown('<div class="hero"><div class="eyebrow">Urban mobility intelligence</div><h1>Command center</h1><p>Monitor transit performance and parking demand from your operational datasets.</p></div>', unsafe_allow_html=True)
    metric_cards(bus, parking)
    if page == "Overview":
        overview(bus, parking)
    elif page == "Bus Analytics":
        bus_analytics(bus)
    elif page == "Parking Analytics":
        parking_analytics(parking)
    else:
        reports(bus, parking)


if __name__ == "__main__":
    main()

    