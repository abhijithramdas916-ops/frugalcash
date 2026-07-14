import pandas as pd
import streamlit as st
import plotly.express as px
import anthropic
import pdfplumber
import io
import json
import re
import time
import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Frugal Cash",
    page_icon="💰",
    layout="wide"
)

# ── Custom styling ───────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem; max-width: 1100px;}
    div[data-testid="stMetric"] {
        background: #F7F9FA;
        border: 1px solid #E4E9EC;
        border-radius: 10px;
        padding: 14px 12px 10px 12px;
    }
    div[data-testid="stMetricLabel"] {font-size: 13px; color: #4A4A4A;}
    h1 {color: #0F2740; font-weight: 700;}
    h2, h3 {color: #0F2740;}
    .stTabs [data-baseweb="tab-list"] {gap: 6px;}
    .stTabs [data-baseweb="tab"] {
        background: #F2F5F6;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #0F766E;
        color: white;
    }
    div[data-testid="stForm"] {
        background: #F7F9FA;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #E4E9EC;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Load API key from Streamlit secrets ───────────────────────
try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    api_key = None

if not api_key:
    st.error("API key not configured. Please contact the app administrator.")
    st.stop()

client = anthropic.Anthropic(api_key=api_key)

# ── Vector DB setup (FAISS) ────────────────────────────────────
VECTOR_DIM = 384  # all-MiniLM-L6-v2 output size
INDEX_PATH = "frugal_cash_index.faiss"
METADATA_PATH = "frugal_cash_metadata.pkl"
LOANS_PATH = "frugal_cash_loans.pkl"

@st.cache_resource
def load_embedding_model():
    """Loaded lazily — only the first time it's actually needed (save or ask),
    not at app startup, so the rest of the UI isn't blocked by the download."""
    return SentenceTransformer("all-MiniLM-L6-v2")

def load_vector_store():
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(VECTOR_DIM)
        metadata = []
    return index, metadata

def save_vector_store(index, metadata):
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index, st.session_state.faiss_metadata = load_vector_store()

def add_transactions_to_vector_store(df, user_id):
    texts = [
        f"{row['Date']} {row['Description']} Rs{row['Amount']} category:{row['Category']} subtype:{row['Subtype']}"
        for _, row in df.iterrows()
    ]
    vectors = load_embedding_model().encode(texts, convert_to_numpy=True).astype("float32")
    st.session_state.faiss_index.add(vectors)
    for _, row in df.iterrows():
        st.session_state.faiss_metadata.append({
            "user_id": user_id,
            "date": str(row["Date"]),
            "description": row["Description"],
            "amount": float(row["Amount"]),
            "category": row["Category"],
            "subtype": row["Subtype"],
        })
    save_vector_store(st.session_state.faiss_index, st.session_state.faiss_metadata)

def query_vector_store(question, user_id, k=15):
    if st.session_state.faiss_index.ntotal == 0:
        return []
    q_vector = load_embedding_model().encode([question], convert_to_numpy=True).astype("float32")
    distances, indices = st.session_state.faiss_index.search(q_vector, k)
    results = []
    for idx in indices[0]:
        if idx == -1 or idx >= len(st.session_state.faiss_metadata):
            continue
        meta = st.session_state.faiss_metadata[idx]
        if meta["user_id"] == user_id:
            results.append(meta)
    return results

# ── Loans store (manual input — R3) ────────────────────────────
def load_loans():
    if os.path.exists(LOANS_PATH):
        with open(LOANS_PATH, "rb") as f:
            return pickle.load(f)
    return []

def save_loans(loans):
    with open(LOANS_PATH, "wb") as f:
        pickle.dump(loans, f)

if "loans" not in st.session_state:
    st.session_state.loans = load_loans()

# ── Helper: Retry on overload ─────────────────────────────────
def call_claude(messages, max_tokens=4000):
    for attempt in range(3):
        try:
            return client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=max_tokens,
                messages=messages
            )
        except Exception as e:
            if "overloaded" in str(e).lower() and attempt < 2:
                st.warning(f"API busy — retrying in 5 seconds (attempt {attempt + 1}/3)...")
                time.sleep(5)
                continue
            raise

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
    user_id = st.text_input("User ID (for demo)", value="demo_user")
    st.divider()
    with st.expander("ℹ️ Supported formats & how to use"):
        st.markdown("✅ CSV · ✅ Excel (.xlsx) · ✅ PDF (text-based)")
        st.markdown("1. Enter your monthly salary")
        st.markdown("2. Upload your bank statement")
        st.markdown("3. See your analysis instantly")
        st.markdown("**CSV/Excel columns required:**")
        st.code("Date, Description, Amount, Type")
        st.markdown("Type must contain **Debit** or **Credit**")

# ── File Upload ───────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your bank statement",
    type=["csv", "xlsx", "pdf"],
    help="Supports CSV, Excel and text-based PDF statements"
)

if uploaded_file is None:
    st.info("Upload your bank statement above to get started")
    st.stop()

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
    message = call_claude([{
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
    }])
    raw = message.content[0].text.strip()
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    return match.group(0) if match else raw

# ── Helper: AI categorise all transactions with subtypes (R2) ─
def categorise_all(df):
    transactions_text = "\n".join([
        f"{i+1}. Description: {row['Description']} | Amount: Rs {row['Amount']}"
        for i, row in df.iterrows()
    ])
    message = call_claude([{
        "role": "user",
        "content": f"""Categorise each Indian bank transaction below into a main category and a subtype.

Main Categories:
- Need
- Greed
- Luxury
- Savings
- Loan
- Investment
- Personal Transfer
- Uncategorised

Subtype rules (ONLY for Savings, Loan, Investment; use "" for all other categories):
- Savings subtype: Cash, FD, RD
- Loan subtype: Home, Educational, Credit Card, Car
- Investment subtype: Gold, SIP, Stocks, Real Estate

Category rules:
- Need: essential expenses (rent, groceries, medical, utilities, school fees, petrol, gas, insurance, electricity, broadband)
- Greed: lifestyle spending (dining, food delivery, streaming, ride hailing, online shopping, entertainment, gym, subscriptions)
- Luxury: premium purchases (high value gadgets, luxury brands, 5 star hotels, business class travel, jewellery)
- Savings: FD, RD, recurring deposits, cash savings transfers (NOT SIP/stocks/gold — those are Investment)
- Loan: EMI payments — identify subtype from description (home loan, education loan, credit card payment, car loan)
- Investment: SIP, mutual funds, stocks, gold purchases, real estate payments
- Personal Transfer: UPI transfers to individuals (person names like ABHIJITH, KUMAR, PRIYA), auto drivers, local vendors, small shops, payment gateways like Razorpay where recipient is unclear, charity donations, family transfers
- Uncategorised: genuinely unclear after applying all above rules

Important rules:
- If description looks like a person's name → Personal Transfer
- If description is a local business, small shop, or auto driver → Personal Transfer
- Google, Netflix, Spotify, Amazon → Greed
- Zepto, Blinkit, BigBasket → Need (grocery delivery)

Return ONLY a JSON array, one object per transaction, in the same order:
[{{"category": "Need", "subtype": ""}}, {{"category": "Loan", "subtype": "Credit Card"}}, ...]

No explanation. No other text. Just the JSON array.

Transactions:
{transactions_text}"""
    }])
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
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
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
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
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
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
        st.success(f"Found {len(df)} debit transactions")
    except Exception as e:
        st.error(f"Could not read Excel: {e}")
        st.stop()

if df is None or df.empty:
    st.error("No transactions found.")
    st.stop()

# ── AI categorise all transactions (R2: category + subtype) ───
with st.spinner("AI is analysing your spending..."):
    categories_json = categorise_all(df)

try:
    parsed = json.loads(categories_json)
    parsed = parsed[:len(df)]
    while len(parsed) < len(df):
        parsed.append({"category": "Uncategorised", "subtype": ""})
    df["Category"] = [p.get("category", "Uncategorised") for p in parsed]
    df["Subtype"] = [p.get("subtype", "") for p in parsed]
except (json.JSONDecodeError, Exception):
    df["Category"] = "Uncategorised"
    df["Subtype"] = ""

# ── Good vs Bad loan classification (deterministic, not AI) ───
GOOD_LOAN_TYPES = {"Home", "Educational"}
BAD_LOAN_TYPES = {"Credit Card", "Car"}

def loan_class(row):
    if row["Category"] != "Loan":
        return ""
    if row["Subtype"] in GOOD_LOAN_TYPES:
        return "Good Loan"
    if row["Subtype"] in BAD_LOAN_TYPES:
        return "Bad Loan"
    return "Other Loan"

df["LoanClass"] = df.apply(loan_class, axis=1)

# ── Store to vector db ──────────────────────────────────────
if st.button("💾 Save this statement to my financial history"):
    with st.spinner("Storing in your financial history..."):
        add_transactions_to_vector_store(df, user_id)
    st.success(f"Saved {len(df)} transactions to your history ({st.session_state.faiss_index.ntotal} total stored)")

# ── Summary numbers ───────────────────────────────────────────
total      = df["Amount"].sum()
need       = df[df["Category"] == "Need"]["Amount"].sum()
greed      = df[df["Category"] == "Greed"]["Amount"].sum()
luxury     = df[df["Category"] == "Luxury"]["Amount"].sum()
savings    = df[df["Category"] == "Savings"]["Amount"].sum()
loan_amt   = df[df["Category"] == "Loan"]["Amount"].sum()
invest_amt = df[df["Category"] == "Investment"]["Amount"].sum()
personal   = df[df["Category"] == "Personal Transfer"]["Amount"].sum()

COLOR_MAP = {
    "Need":              "#2E75B6",
    "Greed":             "#FFA500",
    "Luxury":            "#C00000",
    "Savings":           "#70AD47",
    "Loan":              "#7030A0",
    "Investment":        "#00B0A0",
    "Personal Transfer": "#A6A6A6",
    "Uncategorised":     "#BFBFBF"
}

def calculate_optimised_savings(need_amt, greed_amt, luxury_amt, monthly_salary):
    """T17: freed cash if Greed+Luxury spend were trimmed to the 30% target."""
    if monthly_salary <= 0:
        return 0
    target_greed_luxury = monthly_salary * 0.30
    actual_greed_luxury = greed_amt + luxury_amt
    return max(0, actual_greed_luxury - target_greed_luxury)

def calculate_debt_snowball(loans, extra_monthly_payment=0):
    """T18-T20: snowball bad loans smallest balance first, routing freed EMI forward."""
    bad_loans = sorted(
        [l for l in loans if l["type"] in ("Credit Card", "Car")],
        key=lambda l: l["balance"]
    )
    results = []
    remaining_extra = extra_monthly_payment
    for loan in bad_loans:
        monthly_rate = loan["rate"] / 100 / 12
        balance = loan["balance"]
        payment = loan["emi"] + remaining_extra
        months = 0
        while balance > 0 and months < 600:
            balance = balance * (1 + monthly_rate) - payment
            months += 1
        results.append({"name": loan["name"], "months_to_payoff": months, "payment_used": payment})
        remaining_extra = 0
    return results

INVESTMENT_BENCHMARKS = {
    "SIP":          {"avg_annual_return": 12.0, "target_allocation": 0.20},
    "Stocks":       {"avg_annual_return": 14.0, "target_allocation": 0.15},
    "Gold":         {"avg_annual_return": 8.0,  "target_allocation": 0.10},
    "Real Estate":  {"avg_annual_return": 9.0,  "target_allocation": 0.20},
}

def suggest_investment_allocation(freed_cash, existing_savings, benchmarks):
    investable = freed_cash + existing_savings
    allocation = {asset: round(investable * info["target_allocation"], 2) for asset, info in benchmarks.items()}
    return allocation, investable

# ══════════════════════════════════════════════════════════════
# Tabbed layout
# ══════════════════════════════════════════════════════════════
tab_overview, tab_savings_loans, tab_investments, tab_ask = st.tabs(
    ["📊 Overview", "💰 Savings & Loans", "📈 Investments", "💬 Ask Frugal Cash"]
)

# ── TAB: Overview ───────────────────────────────────────────
with tab_overview:
    st.subheader("Your Spending Summary")
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric("Total", f"Rs{total:,.0f}")
    col2.metric("Need", f"Rs{need:,.0f}", f"{need/total*100:.1f}%", delta_color="off")
    col3.metric("Greed", f"Rs{greed:,.0f}", f"{greed/total*100:.1f}%", delta_color="inverse")
    col4.metric("Luxury", f"Rs{luxury:,.0f}", f"{luxury/total*100:.1f}%", delta_color="inverse")
    col5.metric("Savings", f"Rs{savings:,.0f}", f"{savings/total*100:.1f}%", delta_color="normal")
    col6.metric("Loan", f"Rs{loan_amt:,.0f}", f"{loan_amt/total*100:.1f}%", delta_color="off")
    col7.metric("Investment", f"Rs{invest_amt:,.0f}", f"{invest_amt/total*100:.1f}%", delta_color="normal")

    st.markdown("")
    summary = df.groupby("Category")["Amount"].sum().reset_index()
    fig_pie = px.pie(
        summary, names="Category", values="Amount",
        color="Category", color_discrete_map=COLOR_MAP, hole=0.45
    )
    fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=380)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Budget Health Check (50/30/20 Rule)")
    st.caption("Personal Transfers are excluded from this calculation")
    if salary > 0:
        need_pct = need / salary * 100
        greed_pct = greed / salary * 100
        savings_pct = savings / salary * 100

        def rag(value, green, amber):
            if value <= green: return "🟢"
            elif value <= amber: return "🟡"
            else: return "🔴"

        h1, h2, h3 = st.columns(3)
        h1.metric(f"{rag(need_pct, 50, 60)} Need", f"{need_pct:.1f}%", "Target: below 50%", delta_color="off")
        h2.metric(f"{rag(greed_pct, 30, 40)} Greed", f"{greed_pct:.1f}%", "Target: below 30%", delta_color="off")
        h3.metric(f"{rag(savings_pct, 20, 15)} Savings", f"{savings_pct:.1f}%", "Target: above 20%", delta_color="off")
    else:
        st.info("Enter your salary in the sidebar to see budget health")

    with st.expander("📋 View all transactions"):
        category_filter = st.multiselect(
            "Filter by Category",
            options=df["Category"].unique().tolist(),
            default=df["Category"].unique().tolist()
        )
        filtered_df = df[df["Category"].isin(category_filter)]
        st.dataframe(
            filtered_df[["Date", "Description", "Amount", "Category", "Subtype"]],
            use_container_width=True, hide_index=True
        )

    personal_df = df[df["Category"] == "Personal Transfer"]
    if len(personal_df) > 0:
        with st.expander(f"👤 Personal Transfers — Rs{personal:,.0f}"):
            st.dataframe(personal_df[["Date", "Description", "Amount"]], use_container_width=True, hide_index=True)

    remaining_uncat = df[df["Category"] == "Uncategorised"]
    if len(remaining_uncat) > 0:
        with st.expander(f"❓ Uncategorised — Rs{remaining_uncat['Amount'].sum():,.0f}"):
            st.dataframe(remaining_uncat[["Date", "Description", "Amount"]], use_container_width=True, hide_index=True)

# ── TAB: Savings & Loans (R2 + R3) ──────────────────────────
with tab_savings_loans:
    savings_df = df[df["Category"] == "Savings"]
    loan_df = df[df["Category"] == "Loan"]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Savings Breakdown**")
        if len(savings_df) > 0:
            s_summary = savings_df.groupby("Subtype")["Amount"].sum().reset_index()
            fig_s = px.pie(s_summary, names="Subtype", values="Amount", hole=0.45)
            fig_s.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
            st.plotly_chart(fig_s, use_container_width=True)
        else:
            st.caption("No savings transactions found in this statement")

    with col_b:
        st.markdown("**Loans: Good vs Bad**")
        if len(loan_df) > 0:
            l_summary = loan_df.groupby("LoanClass")["Amount"].sum().reset_index()
            fig_l = px.pie(
                l_summary, names="LoanClass", values="Amount", hole=0.45,
                color="LoanClass",
                color_discrete_map={"Good Loan": "#3EB44A", "Bad Loan": "#C00000", "Other Loan": "#BFBFBF"}
            )
            fig_l.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
            st.plotly_chart(fig_l, use_container_width=True)
        else:
            st.caption("No loan transactions found in this statement")

    st.divider()
    st.subheader("🏦 My Loans")
    st.caption("Add loans manually — statements show EMI payments, not remaining balance or interest rate.")

    with st.form("add_loan"):
        c1, c2, c3, c4, c5 = st.columns(5)
        loan_name = c1.text_input("Loan name", "Home Loan")
        loan_type = c2.selectbox("Loan type", ["Home", "Educational", "Credit Card", "Car"])
        balance = c3.number_input("Balance (Rs)", min_value=0, value=250000, step=1000)
        emi = c4.number_input("EMI (Rs)", min_value=0, value=18000, step=500)
        rate = c5.number_input("Rate (%)", min_value=0.0, value=8.5, step=0.1)
        if st.form_submit_button("Add Loan"):
            st.session_state.loans.append({
                "user_id": user_id, "name": loan_name, "type": loan_type,
                "balance": balance, "emi": emi, "rate": rate,
            })
            save_loans(st.session_state.loans)
            st.success(f"Added {loan_name}")

    user_loan_list = [l for l in st.session_state.loans if l["user_id"] == user_id]
    if user_loan_list:
        st.dataframe(pd.DataFrame(user_loan_list).drop(columns=["user_id"]), use_container_width=True, hide_index=True)

    st.subheader("📉 Debt-Free Roadmap")
    if st.button("Calculate my debt-free timeline"):
        if not user_loan_list:
            st.info("Add at least one loan above first.")
        else:
            freed_cash = calculate_optimised_savings(need, greed, luxury, salary)
            snowball = calculate_debt_snowball(user_loan_list, freed_cash)
            st.write(f"Extra monthly capacity freed from optimised spending: **Rs {freed_cash:,.0f}**")
            if not snowball:
                st.info("No Credit Card or Car loans found — snowball applies to 'bad' loan types only.")
            else:
                for r in snowball:
                    st.write(f"**{r['name']}**: paid off in **{r['months_to_payoff']} months** at Rs {r['payment_used']:,.0f}/month")

# ── TAB: Investments (R2 + R4) ──────────────────────────────
with tab_investments:
    invest_df = df[df["Category"] == "Investment"]
    st.markdown("**Investment Breakdown**")
    if len(invest_df) > 0:
        i_summary = invest_df.groupby("Subtype")["Amount"].sum().reset_index()
        fig_i = px.pie(i_summary, names="Subtype", values="Amount", hole=0.45)
        fig_i.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
        st.plotly_chart(fig_i, use_container_width=True)
    else:
        st.caption("No investment transactions found in this statement")

    st.divider()
    st.subheader("📈 Investment Suggestion")
    st.caption("Benchmarks are static reference figures — refresh periodically, not live market data.")
    if st.button("Suggest my investment allocation"):
        freed_cash = calculate_optimised_savings(need, greed, luxury, salary)
        allocation, investable = suggest_investment_allocation(freed_cash, savings, INVESTMENT_BENCHMARKS)
        st.write(f"Total investable amount: **Rs {investable:,.0f}** (freed cash + existing savings)")
        alloc_df = pd.DataFrame(list(allocation.items()), columns=["Asset", "Suggested Amount"])
        fig_alloc = px.pie(alloc_df, names="Asset", values="Suggested Amount", hole=0.45)
        fig_alloc.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
        st.plotly_chart(fig_alloc, use_container_width=True)
        st.dataframe(alloc_df, use_container_width=True, hide_index=True)

# ── TAB: Ask Frugal Cash (RAG) ──────────────────────────────
with tab_ask:
    st.caption("Ask about your spending history across saved statements")
    question = st.text_input("e.g. What was my highest spending month?")
    if question:
        with st.spinner("Searching your history..."):
            retrieved = query_vector_store(question, user_id)

        if not retrieved:
            st.info("No saved history found yet — save a statement in the Overview tab first.")
        else:
            context_text = "\n".join([
                f"{r['date']} | {r['description']} | Rs{r['amount']} | {r['category']} | {r.get('subtype', '')}"
                for r in retrieved
            ])
            answer = call_claude([{
                "role": "user",
                "content": f"""Here is the user's transaction history:
{context_text}

User's question: {question}

Answer using ONLY the data above. Be specific with numbers. If the data doesn't support an answer, say so."""
            }])
            st.write(answer.content[0].text)
