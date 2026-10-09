import pandas as pd
import streamlit as st

st.set_page_config(page_title="RetainIQ", layout="wide")


@st.cache_data
def load():
    return pd.read_csv("data/customer_scores.csv")


df = load()

st.title("RetainIQ: Customer Retention Dashboard")
st.caption("Online Retail II | cutoff 9 Sep 2011 | 90-day churn window | values in £")

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
all_segments = sorted(df["segment_name"].unique())
segments = st.sidebar.multiselect("Segment", all_segments, default=all_segments)
min_prob = st.sidebar.slider("Minimum churn probability", 0.0, 1.0, 0.0, 0.05)
view = df[df["segment_name"].isin(segments) & (df["churn_prob"] >= min_prob)]

if view.empty:
    st.warning("No customers match these filters.")
    st.stop()

# ---------- KPI cards ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(view):,}")
c2.metric("Revenue at risk (90 days)", f"£{view['revenue_at_risk'].sum():,.0f}")
c3.metric("Avg churn probability", f"{view['churn_prob'].mean():.0%}")
c4.metric("High-risk customers (>70%)", f"{(view['churn_prob'] > 0.7).sum():,}")

# ---------- Segments ----------
seg = view.groupby("segment_name").agg(
    customers=("CustomerID", "count"),
    avg_churn_prob=("churn_prob", "mean"),
    revenue_at_risk=("revenue_at_risk", "sum"),
).round(2)

left, right = st.columns(2)
with left:
    st.subheader("Revenue at risk by segment")
    st.bar_chart(seg["revenue_at_risk"])
with right:
    st.subheader("Segment summary")
    st.dataframe(seg.sort_values("revenue_at_risk", ascending=False))

# ---------- Priority list ----------
st.subheader("Who to contact first")
top_n = st.slider("Number of customers", 10, 200, 50, 10)
priority = (
    view.sort_values("revenue_at_risk", ascending=False)
    .head(top_n)[["CustomerID", "segment_name", "churn_prob", "value_90d", "revenue_at_risk"]]
    .round(2)
)
st.dataframe(priority)
st.download_button("Download call list (CSV)", priority.to_csv(index=False),
                   "priority_customers.csv", "text/csv")

# ---------- Campaign simulator ----------
st.subheader("Campaign simulator (uses all customers, simulated)")
a, b, c = st.columns(3)
n = a.slider("Customers to contact", 100, 1000, 500, 50)
effect = b.slider("Assumed churn reduction", 0.05, 0.50, 0.20, 0.05)
cost = c.slider("Cost per contact (£)", 1, 20, 5)

saved_best = df.nlargest(n, "revenue_at_risk")["revenue_at_risk"].sum() * effect
saved_prob = df.nlargest(n, "churn_prob")["revenue_at_risk"].sum() * effect
saved_rand = df["revenue_at_risk"].mean() * n * effect   # expected value of a random pick

m1, m2, m3, m4 = st.columns(4)
m1.metric("Top by revenue at risk", f"£{saved_best:,.0f}")
m2.metric("Top by churn probability", f"£{saved_prob:,.0f}")
m3.metric("Random pick", f"£{saved_rand:,.0f}")
m4.metric("Net gain (best strategy)", f"£{saved_best - n * cost:,.0f}")

st.bar_chart(pd.DataFrame(
    {"Revenue saved (£)": [saved_best, saved_prob, saved_rand]},
    index=["By revenue at risk", "By churn probability", "Random"]))
st.caption("The offer effect and contact cost are assumptions. "
           "Real uplift needs an A/B test with a control group.")
