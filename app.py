import pandas as pd
import streamlit as st
import plotly.express as px
import anthropic
import pdfplumber
import io
import json
import re

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Frugal Cash",
    page_icon="💰",
    layout="wide"
)

# ── Load API key from Streamlit secrets ───────────────────────
try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    api_key = None

if not api_key:
    st.error("API key not configured. Please contact the app administrator.")
    st.stop()

client = anthropic.Anthropic(api_key=api_key)

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
        step=1000
    )
    st.divider()
    st.markdown("### Supported formats")
    st.markdown("✅ CSV")
    st.markdown("✅ Excel (.xlsx)")
    st.markdown("✅ PDF (text-based)")
    st.divider()
    st.markdown("### How to use")
    st.markdown("1. Enter your monthly salary")
    st.markdown("2. Upload your bank statement")
    st.markdown("3. See your analysis instantly")
    st.divider()
    st.markdown("### CSV / Excel Format Required")
    st.code("Date, Description, Amount, Type")
    st.markdown("Type must contain **Debit** or **Credit**")
    st.markdown("### PDF")
    st.markdown("Any text-based bank statement PDF works directly — no formatting needed")

# ── File Upload ───────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your bank statement",
    type=["csv", "xlsx", "pdf"],
    help="Supports CSV, Excel and text-based PDF statements"
)

if uploaded_file is None:
    st.info("Upload your bank statement above to get started")
    st.stop()

# ── Helper: Clean AI JSON response ───────────────────────────
def clean_json(raw):
    raw = raw.strip()
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    return match.group(0) if match else raw

# ── Helper: Extract text from PDF ────────────────────────────
def extract_pdf_text(file):
    text = ""
    with pdfplumber.open(io.BytesIO(file.read())) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ── Helper: Extract transactions from PDF using AI ────────────
def extract_transactions(raw_text):
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": f"""This is raw text from an Indian bank statement.

Your job:
1. Find ALL debit transactions (money going OUT)
2. Ignore credit transactions (salary, deposits, refunds)
3. Ignore headers, footers, account details, balance rows
4. Return ONLY a valid JSON array like this:

[
  {{"Date": "01-Oct-2024", "Description": "SWIGGY ORDER", "Amount": 350.00}},
  {{"Date": "03-Oct-2024", "Description": "BESCOM ELECTRICITY", "Amount": 1200.00}}
]

Rules:
- Amount must be a number only (no Rs symbol, no commas)
- Return ONLY the JSON array. No explanation. No other text.
- If no debit transactions found, return []

Bank statement text:
{raw_text}"""
        }]
    )
    raw = message.content[0].text.strip()
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    return match.group(0) if match else raw

# ── Helper: AI categorise all transactions in one call ────────
def categorise_all(df):
    transactions_text = "\n".join([
        f"{i+1}. Description: {row['Description']} | Amount: Rs {row['Amount']}"
        for i, row in df.iterrows()
    ])
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": f"""Categorise each Indian bank transaction below.

Categories:
- Need: essential expenses (rent, groceries, EMI, medical, utilities, school fees, petrol, gas, insurance)
- Greed: lifestyle spending (dining, food delivery, streaming, ride hailing, shopping, entertainment, gym)
- Luxury: premium purchases (high value gadgets, luxury brands, 5 star hotels, business class travel)
- Savings: investments (SIP, FD, mutual funds, stocks, gold, PPF, NPS)
- Uncategorised: genuinely unclear

Return ONLY a JSON array with one category per transaction in the same order:
["Need", "Greed", "Luxury", "Savings", "Uncategorised", ...]

No explanation. No other text. Just the JSON array.

Transactions:
{transactions_text}"""
        }]
    )
    raw = message.content[0].text.strip()
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    return match.group(0) if match else raw

# ── Load file ─────────────────────────────────────────────────
file_type = uploaded_file.name.split(".")[-1].lower()
df = None

if file_type == "pdf":
    with st.spinner("Reading your PDF statement..."):
        raw_text = extract_pdf_text(uploaded_file)

    if not raw_text.strip():
        st.error("Could not extract text from this PDF.")
        st.stop()

    with st.spinner("AI is extracting your debit transactions..."):
        json_result = extract_transactions(raw_text)

    try:
        transactions = json.loads(json_result)
        if not transactions:
            st.error("No debit transactions found in this PDF.")
            st.stop()
        df = pd.DataFrame(transactions)
        df["Amount"] = pd.to_numeric(
            df["Amount"], errors="coerce"
        ).fillna(0)
        st.success(f"Found {len(df)} debit transactions")
    except json.JSONDecodeError:
        st.error("Could not parse the PDF. The statement format may not be supported yet.")
        st.stop()

elif file_type == "csv":
    try:
        df_raw = pd.read_csv(uploaded_file)
        required = ["Date", "Description", "Amount", "Type"]
        missing = [c for c in required if c not in df_raw.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()
        df = df_raw[df_raw["Type"] == "Debit"].copy()
        df["Amount"] = pd.to_numeric(
            df["Amount"], errors="coerce"
        ).fillna(0)
        st.success(f"Found {len(df)} debit transactions")
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

elif file_type == "xlsx":
    try:
        df_raw = pd.read_excel(uploaded_file)
        required = ["Date", "Description", "Amount", "Type"]
        missing = [c for c in required if c not in df_raw.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()
        df = df_raw[df_raw["Type"] == "Debit"].copy()
        df["Amount"] = pd.to_numeric(
            df["Amount"], errors="coerce"
        ).fillna(0)
        st.success(f"Found {len(df)} debit transactions")
    except Exception as e:
        st.error(f"Could not read Excel: {e}")
        st.stop()

if df is None or df.empty:
    st.error("No transactions found.")
    st.stop()

# ── AI categorise all transactions ────────────────────────────
with st.spinner("AI is analysing your spending..."):
    categories_json = categorise_all(df)

try:
    categories = json.loads(categories_json)
    if len(categories) != len(df):
        categories = categories + ["Uncategorised"] * (
            len(df) - len(categories)
        )
    df["Category"] = categories
except json.JSONDecodeError:
    df["Category"] = "Uncategorised"

# ── Summary numbers ───────────────────────────────────────────
total   = df["Amount"].sum()
need    = df[df["Category"] == "Need"]["Amount"].sum()
greed   = df[df["Category"] == "Greed"]["Amount"].sum()
luxury  = df[df["Category"] == "Luxury"]["Amount"].sum()
savings = df[df["Category"] == "Savings"]["Amount"].sum()
uncat   = df[df["Category"] == "Uncategorised"]["Amount"].sum()

# ── Metric cards ──────────────────────────────────────────────
st.subheader("Your Spending Summary")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Spent", f"Rs{total:,.0f}")
col2.metric("Need",        f"Rs{need:,.0f}",
            f"{need/total*100:.1f}%",    delta_color="off")
col3.metric("Greed",       f"Rs{greed:,.0f}",
            f"{greed/total*100:.1f}%",   delta_color="inverse")
col4.metric("Luxury",      f"Rs{luxury:,.0f}",
            f"{luxury/total*100:.1f}%",  delta_color="inverse")
col5.metric("Savings",     f"Rs{savings:,.0f}",
            f"{savings/total*100:.1f}%", delta_color="normal")

st.divider()

# ── Charts ────────────────────────────────────────────────────
COLOR_MAP = {
    "Need":          "#2E75B6",
    "Greed":         "#FFA500",
    "Luxury":        "#C00000",
    "Savings":       "#70AD47",
    "Uncategorised": "#BFBFBF"
}

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Spending Breakdown")
    summary = df.groupby("Category")["Amount"].sum().reset_index()
    fig_pie = px.pie(
        summary, names="Category", values="Amount",
        color="Category", color_discrete_map=COLOR_MAP,
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Category Comparison")
    fig_bar = px.bar(
        summary, x="Category", y="Amount",
        color="Category", color_discrete_map=COLOR_MAP,
        text_auto=True
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── Budget Health Check ───────────────────────────────────────
st.subheader("Budget Health Check (50/30/20 Rule)")

if salary > 0:
    need_pct    = need    / salary * 100
    greed_pct   = greed   / salary * 100
    savings_pct = savings / salary * 100

    def rag(value, green, amber):
        if value <= green:   return "Green"
        elif value <= amber: return "Amber"
        else:                return "Red"

    h1, h2, h3 = st.columns(3)
    h1.metric(
        f"{rag(need_pct, 50, 60)} Need",
        f"{need_pct:.1f}% of income",
        "Target: below 50%",
        delta_color="off"
    )
    h2.metric(
        f"{rag(greed_pct, 30, 40)} Greed",
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
    st.info("Enter your salary in the sidebar to see budget health")

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
remaining_uncat = df[df["Category"] == "Uncategorised"]
if len(remaining_uncat) > 0:
    st.divider()
    st.subheader("Uncategorised Transactions")
    st.caption(
        f"Rs{remaining_uncat['Amount'].sum():,.0f} could not be categorised."
    )
    st.dataframe(
        remaining_uncat[["Date", "Description", "Amount"]],
        use_container_width=True,
        hide_index=True
                           )
