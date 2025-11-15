# ============================================================
# Script: generate_engagement_facts.py
# Purpose: Generate a cleaned engagement_facts.csv dataset
#          for chatbot embedding, from your Excel content data.
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# 1️⃣ File paths
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CONTENT_PATH = BASE_DIR / "acies-global_content.xlsx"
FOLLOWERS_PATH = BASE_DIR / "acies-global_followers.xls"
OUTPUT_DIR = BASE_DIR / "chatbot_data"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "engagement_facts.csv"

# ------------------------------------------------------------
# 2️⃣ Load content file
# ------------------------------------------------------------
try:
    df = pd.read_excel(CONTENT_PATH, engine="openpyxl")
except Exception as e:
    raise SystemExit(f"❌ Could not read content file: {e}")

# Prefer “All posts” sheet if available
try:
    xl = pd.ExcelFile(CONTENT_PATH, engine="openpyxl")
    if "All posts" in xl.sheet_names:
        df = pd.read_excel(CONTENT_PATH, sheet_name="All posts", engine="openpyxl")
except Exception:
    pass

# ------------------------------------------------------------
# 3️⃣ Auto-detect column names safely
# ------------------------------------------------------------
def find_col(df, keywords):
    for c in df.columns:
        if all(k.lower() in c.lower() for k in keywords):
            return c
    return None

eng_col = find_col(df, ["engagement", "rate"])
imp_col = find_col(df, ["impression"])
clicks_col = find_col(df, ["click"])
likes_col = find_col(df, ["like"])
comments_col = find_col(df, ["comment"])
reposts_col = find_col(df, ["repost"])
follows_col = find_col(df, ["follow"])
ctype_col = find_col(df, ["content", "type"])
title_col = find_col(df, ["post", "title"])
hashtags_col = find_col(df, ["hashtag"])
created_col = find_col(df, ["created", "date"])

missing = [k for k, v in {
    "Engagement Rate": eng_col,
    "Impressions": imp_col,
    "Created Date": created_col
}.items() if v is None]

if missing:
    raise SystemExit(f"❌ Missing essential columns: {', '.join(missing)}")

# ------------------------------------------------------------
# 4️⃣ Clean + prepare data
# ------------------------------------------------------------
df[eng_col] = pd.to_numeric(df[eng_col], errors="coerce")
df[imp_col] = pd.to_numeric(df[imp_col], errors="coerce")

# Derived metrics
df["Total Engagement"] = df[[x for x in [clicks_col, likes_col, comments_col, reposts_col, follows_col] if x in df.columns]].sum(axis=1)
df["Engagement Rate (%)"] = df[eng_col] * 100 if df[eng_col].mean() < 1 else df[eng_col]  # normalize if needed

avg_er = df["Engagement Rate (%)"].mean(skipna=True)
total_eng = df["Total Engagement"].sum(skipna=True)
total_impr = df[imp_col].sum(skipna=True)
post_count = df.shape[0]

print("✅ Engagement Rate Detected Column:", eng_col)
print(f"📊 Avg Engagement Rate = {avg_er:.2f}%")
print(f"📈 Total Engagements = {int(total_eng):,}")
print(f"👀 Total Impressions = {int(total_impr):,}")
print(f"📝 Total Posts = {post_count}")

# ------------------------------------------------------------
# 5️⃣ Create structured facts for chatbot
# ------------------------------------------------------------
facts = []
for _, row in df.iterrows():
    facts.append({
        "Post Title": str(row.get(title_col, "")),
        "Content Type": str(row.get(ctype_col, "")),
        "Engagement Rate (%)": round(row.get("Engagement Rate (%)", np.nan), 2),
        "Total Engagement": int(row.get("Total Engagement", 0)),
        "Impressions": int(row.get(imp_col, 0)),
        "Hashtags": str(row.get(hashtags_col, "")),
        "Created Date": str(row.get(created_col, "")),
    })

facts_df = pd.DataFrame(facts)

# ------------------------------------------------------------
# 6️⃣ Save to CSV
# ------------------------------------------------------------
facts_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"\n✅ engagement_facts.csv generated successfully!")
print(f"📁 Saved at: {OUTPUT_PATH}")
