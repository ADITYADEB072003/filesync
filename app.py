import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import PyPDF2
import re
from datetime import datetime

st.set_page_config(layout="wide")
plt.rcParams.update({'figure.autolayout': True})

# ----------------------
# CATEGORY KEYWORDS
# ----------------------
CATEGORY_KEYWORDS = {
    "Food & Delivery": ["zomato", "swiggy", "blinkit", "fat belly", "restaurant", "food", "amazon pay"],
    "Subscriptions": ["youtube", "google", "apple services", "apple", "jio", "airtel", "subscription"],
    "Utilities": ["tangedco", "electric", "water", "gas", "recharge", "jio fiber", "airtel postpaid"],
    "Shopping": ["amazon", "flipkart", "store", "mall", "neofinity"],
    "Transfers": ["upi", "neft", "rtgs", "imps", "transfer", "paytm", "gpay", "razorpay"],
    "Healthcare": ["clinic", "hospital", "mediplus", "dental"],
    "Cash Withdrawal": ["atm", "cash withdrawal", "cwdr"],
    "Other": []
}


# ----------------------
# Helper functions
# ----------------------
def extract_text_from_pdf(path):
    text = ""
    reader = PyPDF2.PdfReader(path)
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text


def normalize_amount(value):
    if value is None:
        return None
    v = value.replace(",", "")
    try:
        return float(v)
    except:
        return None


def detect_date(line):
    patterns = [
        r"\b(\d{2}/\d{2}/\d{2,4})\b",
        r"\b(\d{2}-\d{2}-\d{2,4})\b",
    ]
    for pat in patterns:
        m = re.search(pat, line)
        if m:
            raw = m.group(1)
            fmts = ["%d/%m/%y", "%d/%m/%Y", "%d-%m-%y", "%d-%m-%Y"]
            for fmt in fmts:
                try:
                    return datetime.strptime(raw, fmt)
                except:
                    pass
    return None


def assign_category(description):
    desc = description.lower()
    for cat, keys in CATEGORY_KEYWORDS.items():
        for k in keys:
            if k in desc:
                return cat
    return "Other"


def parse_statement(text):
    lines = text.split("\n")
    rows = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        date = detect_date(line)
        if not date:
            continue  # only process lines with dates

        floats = re.findall(r"[-]?\d+\.\d{2}", line)
        floats = [normalize_amount(x) for x in floats]

        debit = credit = balance = None

        if len(floats) >= 3:
            debit, credit, balance = floats[-3], floats[-2], floats[-1]
        elif len(floats) == 2:
            debit, balance = floats
        elif len(floats) == 1:
            debit = floats[0]

        description = re.sub(r"\d{2}[\/\-]\d{2}[\/\-]\d{2,4}", "", line)
        description = re.sub(r"[-]?\d+\.\d{2}", "", description)
        description = " ".join(description.split())

        rows.append({
            "Date": date,
            "Description": description,
            "Debit": debit,
            "Credit": credit,
            "Balance": balance,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["Debit"] = pd.to_numeric(df["Debit"], errors="coerce")
    df["Credit"] = pd.to_numeric(df["Credit"], errors="coerce")
    df["Balance"] = pd.to_numeric(df["Balance"], errors="coerce")

    df["Type"] = df.apply(lambda r: "Debit" if r["Debit"] else ("Credit" if r["Credit"] else "Other"), axis=1)
    df["Amount"] = df["Debit"].fillna(0) + df["Credit"].fillna(0)
    df["Category"] = df["Description"].apply(assign_category)

    return df


# ----------------------
# Dashboard UI
# ----------------------
st.title("📊 Bank Statement Dashboard (PDF Auto-Report)")

uploaded = st.file_uploader("Upload your bank statement (PDF)", type=["pdf"])

if uploaded:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        pdf_path = tmp.name

    st.info("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)

    st.info("Parsing transactions...")
    df = parse_statement(text)

    if df.empty:
        st.error("Unable to extract transactions. The statement format may need customized parsing.")
        st.stop()

    st.success(f"Parsed {len(df)} transactions!")
    st.dataframe(df)

    # Download CSV
    st.download_button("Download transactions.csv", df.to_csv(index=False), "transactions.csv")

    # -------------- CHART 1: Monthly Spending -----------------
    df["Month"] = df["Date"].dt.to_period("M")
    monthly = df[df["Type"] == "Debit"].groupby("Month")["Debit"].sum()

    fig1, ax1 = plt.subplots()
    ax1.plot(monthly.index.astype(str), monthly.values, marker="o")
    ax1.set_title("Monthly Spending Trend")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Total Debits")
    ax1.grid(True)
    st.pyplot(fig1)

    # -------------- CHART 2: Income vs Expense -----------------
    agg = df.groupby(["Month", "Type"])["Amount"].sum().unstack(fill_value=0)

    fig2, ax2 = plt.subplots()
    agg.plot(kind="bar", ax=ax2)
    ax2.set_title("Income vs Expense (Monthly)")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Amount")
    st.pyplot(fig2)

    # -------------- CHART 3: Category Pie Chart -----------------
    cat = df[df["Type"] == "Debit"].groupby("Category")["Debit"].sum()

    fig3, ax3 = plt.subplots()
    if not cat.empty:
        ax3.pie(cat.values, labels=cat.index, autopct="%1.1f%%")
    ax3.set_title("Spending by Category")
    st.pyplot(fig3)

    # -------------- CHART 4: Top Payees -----------------
    df["Payee"] = df["Description"].str[:40]

    top_payees = df[df["Type"] == "Debit"].groupby("Payee")["Debit"].sum().sort_values(ascending=False).head(12)

    fig4, ax4 = plt.subplots()
    top_payees.plot(kind="barh", ax=ax4)
    ax4.set_title("Top Payees")
    ax4.set_xlabel("Amount")
    st.pyplot(fig4)

    st.success("Dashboard generated successfully!")