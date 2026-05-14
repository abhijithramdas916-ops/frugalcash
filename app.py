
import pandas as pd
import streamlit as st
import plotly.express as px

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Frugal Cash",
    page_icon="💰",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────
st.title("💰 Frugal Cash")
st.caption("Upload your bank statement and instantly know where your money goes")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    salary = st.number_input(
        "Your Monthly Salary (Rs)",
        min_value=0,
        value=85000,
        step=1000,
        help="Used to calculate your 50/30/20 budget health"
    )
    st.divider()
    st.markdown("### How to use")
    st.markdown("1. Enter your monthly salary above")
    st.markdown("2. Upload your bank statement CSV")
    st.markdown("3. See your spending analysis instantly")
    st.divider()
    st.markdown("### CSV Format Required")
    st.markdown("Your CSV must have these columns:")
    st.code("Date, Description, Amount, Type")
    st.markdown("Type column must contain **Debit** or **Credit**")

# ── File Upload ───────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your bank statement (CSV or Excel)",
    type=["csv", "xlsx"],
    help="Download your bank statement from net banking and upload here"
)

# ── Process only if file is uploaded ─────────────────────────
if uploaded_file is None:
    st.info("Upload your bank statement above to get started")
    st.stop()

# ── Load the file ─────────────────────────────────────────────
try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

# ── Validate columns ──────────────────────────────────────────
required_columns = ["Date", "Description", "Amount", "Type"]
missing = [c for c in required_columns if c not in df.columns]
if missing:
    st.error(f"Missing columns in your file: {missing}")
    st.markdown("Please make sure your CSV has: **Date, Description, Amount, Type**")
    st.stop()

# ── Filter only Debits ────────────────────────────────────────
df = df[df["Type"] == "Debit"].copy()

if df.empty:
    st.warning("No Debit transactions found in this file.")
    st.stop()

# ── Keyword lists ─────────────────────────────────────────────
NEED_KEYWORDS = [
    "RENT", "ELECTRICITY", "BESCOM", "GROCERY", "GROFERS", "DMART",
    "MEDICAL", "PHARMACY", "SCHOOL", "PETROL", "LOAN", "EMI",
    "INSURANCE", "WATER", "GAS", "MOBILE", "INTERNET", "BROADBAND"
]
GREED_KEYWORDS = [
    "SWIGGY", "ZOMATO", "UBER", "NETFLIX", "GYM", "MOVIE",
    "STARBUCKS", "COFFEE", "AMAZON", "FLIPKART", "MYNTRA",
    "DINING", "RESTAURANT", "CAFE", "SPOTIFY", "PRIME"
]
LUXURY_KEYWORDS = [
    "LOUIS VUITTON", "APPLE", "IPHONE", "LUXURY", "HOTEL",
    "RESORT", "GUCCI", "PRADA", "BUSINESS CLASS", "ROLEX"
]
SAVINGS_KEYWORDS = [
    "SIP", "MUTUAL FUND", "INVESTMENT", "STOCK", "ZERODHA",
    "GROWW", "PPF", "NPS", "FD", "RD", "GOLD"
]

# ── Categorise ────────────────────────────────────────────────
def categorise(description):
    description = str(description).upper()
    for keyword in LUXURY_KEYWORDS:
        if keyword in description:
            return "Luxury"
    for keyword in SAVINGS_KEYWORDS:
        if keyword in description:
            return "Savings"
    for keyword in NEED_KEYWORDS:
        if keyword in description:
            return "Need"
    for keyword in GREED_KEYWORDS:
        if keyword in description:
            return "Greed"
    return "Uncategorised"

df["Category"] = df["Description"].apply(categorise)
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)

# ── Summary numbers ───────────────────────────────────────────
total   = df["Amount"].sum()
need    = df[df["Category"] == "Need"]["Amount"].sum()
greed   = df[df["Category"] == "Greed"]["Amount"].sum()
luxury  = df[df["Category"] == "Luxury"]["Amount"].sum()
savings = df[df["Category"] == "Savings"]["Amount"].sum()
uncat   = df[df["Category"] == "Uncategorised"]["Amount"].sum()

# ── Metric cards ──────────────────────────────────────────────
st.subheader("Spending Summary")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Spent",  f"Rs{total:,.0f}")
col2.metric("Need",         f"Rs{need:,.0f}",
            f"{need/total*100:.1f}%",    delta_color="off")
col3.metric("Greed",        f"Rs{greed:,.0f}",
            f"{greed/total*100:.1f}%",   delta_color="inverse")
col4.metric("Luxury",       f"Rs{luxury:,.0f}",
            f"{luxury/total*100:.1f}%",  delta_color="inverse")
col5.metric("Savings",      f"Rs{savings:,.0f}",
            f"{savings/total*100:.1f}%", delta_color="normal")

st.divider()

# ── Charts ────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Spending Breakdown")
    summary = df.groupby("Category")["Amount"].sum().reset_index()
    fig_pie = px.pie(
        summary,
        names="Category",
        values="Amount",
        color="Category",
        color_discrete_map={
            "Need":          "#2E75B6",
            "Greed":         "#FFA500",
            "Luxury":        "#C00000",
            "Savings":       "#70AD47",
            "Uncategorised": "#BFBFBF"
        },
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Category Comparison")
    fig_bar = px.bar(
        summary,
        x="Category",
        y="Amount",
        color="Category",
        color_discrete_map={
            "Need":          "#2E75B6",
            "Greed":         "#FFA500",
            "Luxury":        "#C00000",
            "Savings":       "#70AD47",
            "Uncategorised": "#BFBFBF"
        },
        text_auto=True
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Budget Health Check (50/30/20) ────────────────────────────
st.subheader("Budget Health Check (50/30/20 Rule)")

if salary > 0:
    need_pct    = need    / salary * 100
    greed_pct   = greed   / salary * 100
    savings_pct = savings / salary * 100

    def rag(value, green, amber):
        if value <= green:
            return "Green"
        elif value <= amber:
            return "Amber"
        else:
            return "Red"

    h1, h2, h3 = st.columns(3)
    h1.metric(
        f"{rag(need_pct, 50, 60)} Need Spending",
        f"{need_pct:.1f}% of income",
        "Target: below 50%",
        delta_color="off"
    )
    h2.metric(
        f"{rag(greed_pct, 30, 40)} Greed Spending",
        f"{greed_pct:.1f}% of income",
        "Target: below 30%",
        delta_color="off"
    )
    h3.metric(
        f"{rag(savings_pct, 20, 15)} Savings",
        f"{savings_pct:.1f}% of income",
        "Target: above 20%",
        delta_color="off"
    )
else:
    st.info("Enter your monthly salary in the sidebar to see budget health")

st.divider()

# ── Transaction table ─────────────────────────────────────────
st.subheader("All Transactions")

category_filter = st.multiselect(
    "Filter by Category",
    options=df["Category"].unique().tolist(),
    default=df["Category"].unique().tolist()
)

filtered_df = df[df["Category"].isin(category_filter)]

st.dataframe(
    filtered_df[["Date", "Description", "Amount", "Category"]],
    use_container_width=True,
    hide_index=True
)

# ── Uncategorised helper ──────────────────────────────────────
if uncat > 0:
    st.divider()
    st.subheader("Uncategorised Transactions")
    st.caption(
        f"Rs{uncat:,.0f} could not be categorised. "
        "Add these merchant names to your keyword list to improve accuracy."
    )
    st.dataframe(
        df[df["Category"] == "Uncategorised"][["Date", "Description", "Amount"]],
        use_container_width=True,
        hide_index=True
    )
