import re
import pandas as pd
import numpy as np

INPUT_CSV = '/mnt/data/Acct Statement_XX5660_29112025.csv'  # change to your file path
OUTPUT_CSV = '/mnt/data/extracted_statement.csv'

# --- Helpers ---
date_re = re.compile(r'^\s*\d{1,2}/\d{1,2}/\d{2,4}\s*$')  # simple dd/mm/yy or d/m/yyyy
header_keywords = {
    'deposit': ['deposit', 'deposit amt', 'cr', 'credit'],
    'withdrawal': ['withdrawal', 'withdrawal amt', 'dr', 'debit'],
    'closing': ['closing balance', 'closing', 'balance']
}

def looks_like_date(s):
    if pd.isna(s): 
        return False
    return bool(date_re.match(str(s).strip()))

def numeric_count(series):
    # try convert to numeric after cleaning commas and spaces
    s = series.astype(str).str.replace(',', '').str.strip()
    coerced = pd.to_numeric(s, errors='coerce')
    return coerced.notna().sum()

def clean_numeric(series):
    s = series.astype(str).str.replace(',', '').str.strip()
    return pd.to_numeric(s, errors='coerce')

# --- Read CSV with no header (preserve rows exactly as in file) ---
df_raw = pd.read_csv(INPUT_CSV, header=None, dtype=str, keep_default_na=False)

# --- Find first row that looks like a date anywhere in the row ---
date_row_idx = None
for i, row in df_raw.iterrows():
    if any(looks_like_date(cell) for cell in row):
        date_row_idx = i
        break

if date_row_idx is None:
    raise SystemExit("Could not find a row containing a date. Check the CSV format or adjust 'date_re'.")

# --- Try to detect textual header row just above the date row (if present) ---
header_row_idx = None
if date_row_idx - 1 >= 0:
    possible_header = df_raw.loc[date_row_idx - 1].astype(str).str.lower().tolist()
    # check for header keywords
    if any(any(k in (cell or '') for k in header_keywords['deposit']) for cell in possible_header) \
       or any(any(k in (cell or '') for k in header_keywords['closing']) for cell in possible_header):
        header_row_idx = date_row_idx - 1

# --- Build data frame from date row onwards, optionally use detected header ----
if header_row_idx is not None:
    df_data = pd.read_csv(INPUT_CSV, header=header_row_idx, skiprows=range(0, header_row_idx), dtype=str)
else:
    # No textual header: read as header=None then slice from date row and set generic column names
    df_temp = pd.read_csv(INPUT_CSV, header=None, dtype=str)
    df_data = df_temp.loc[date_row_idx:].reset_index(drop=True)
    df_data.columns = [f'col_{i}' for i in range(df_data.shape[1])]

# --- If header detected, ensure all columns exist as strings ---
df_data = df_data.astype(str)

# --- Identify candidate numeric columns by numeric counts ---
numeric_counts = {col: numeric_count(df_data[col]) for col in df_data.columns}
# sort columns by index order (appearance) but we also want the ones with most numeric values
sorted_by_index = list(df_data.columns)
sorted_numeric = sorted(numeric_counts.items(), key=lambda x: (-x[1], sorted_by_index.index(x[0])))

# If header row provided, try to map by header names
withdraw_col = deposit_col = closing_col = None
if header_row_idx is not None:
    lower_cols = {col: col.lower() for col in df_data.columns}
    # find deposit
    for col, low in lower_cols.items():
        if any(k in low for k in header_keywords['deposit']):
            deposit_col = col
            break
    for col, low in lower_cols.items():
        if any(k in low for k in header_keywords['withdrawal']):
            withdraw_col = col
            break
    for col, low in lower_cols.items():
        if any(k in low for k in header_keywords['closing']):
            closing_col = col
            break

# Fallback: pick three best numeric columns (left-to-right) if any are still None
if not all([withdraw_col, deposit_col, closing_col]):
    # choose top 3 numeric columns by count; if less than 3 available, pick what exists
    top_numeric_cols = [c for c, cnt in sorted_numeric if cnt > 0]
    # prefer column order (left to right)
    top_numeric_cols = sorted(top_numeric_cols, key=lambda c: sorted_by_index.index(c))
    # take first three
    if len(top_numeric_cols) >= 3:
        withdraw_col, deposit_col, closing_col = top_numeric_cols[:3]
    else:
        # if less than 3 numeric columns, attempt to still assign
        if len(top_numeric_cols) == 2:
            withdraw_col, deposit_col = top_numeric_cols
        elif len(top_numeric_cols) == 1:
            withdraw_col = top_numeric_cols[0]

# --- Build output DataFrame with standard columns ---
out = pd.DataFrame()
# Try to include a date/description column if present (a column that looks like dates in at least one cell)
date_col = None
for col in df_data.columns:
    if df_data[col].apply(lambda x: bool(date_re.match(str(x).strip()))).any():
        date_col = col
        break

if date_col:
    out['Date/Info'] = df_data[date_col].replace('', np.nan)

# Add numeric columns (cleaned)
if withdraw_col:
    out['Withdrawal'] = clean_numeric(df_data[withdraw_col])
else:
    out['Withdrawal'] = np.nan

if deposit_col:
    out['Deposit'] = clean_numeric(df_data[deposit_col])
else:
    out['Deposit'] = np.nan

if closing_col:
    out['Closing Balance'] = clean_numeric(df_data[closing_col])
else:
    out['Closing Balance'] = np.nan

# Drop rows that are essentially empty (no numeric values)
out = out.dropna(how='all', subset=['Withdrawal','Deposit','Closing Balance'])

# Reset index and save
out = out.reset_index(drop=True)
out.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote extracted file to: {OUTPUT_CSV}")
print("Preview:")
print(out.head(10).to_string(index=False))