# =====================================================
# Streamlit App: Engagement Intelligence (Acies)
# =====================================================

import os
import re
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
from nltk.corpus import stopwords
from collections import Counter

# Optional imports
try:
    from sentence_transformers import SentenceTransformer
except:
    SentenceTransformer = None
try:
    from langchain_community.vectorstores import FAISS
except:
    FAISS = None
try:
    from langchain_groq import ChatGroq
except:
    ChatGroq = None
try:
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain.embeddings.base import Embeddings
except:
    RetrievalQA = None
    PromptTemplate = None
    Embeddings = None

# BASIC SETUP
st.set_page_config(page_title="Engagement Intelligence", layout="wide")
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

CONTENT_PATH = r"C:\Users\DELL\Downloads\archive (2)\acies-global_content.xlsx"
FOLLOWERS_PATH = r"C:\Users\DELL\Downloads\archive (2)\acies-global_followers.xls"

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "models" / "vector_store"

GROQ_KEY = st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY")

# ========== HELPER FUNCTIONS ==========
def read_excel_safe(path: str):
    p = Path(path)
    if not p.exists():
        st.error(f"❌ File not found: {path}")
        st.stop()
    if p.suffix == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    return pd.read_excel(path, engine="xlrd")

def to_datetime_col(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def derive_time_columns(df):
    for base in ["Created date", "Date", "TimeStamp"]:
        if base in df.columns:
            df = to_datetime_col(df, base)
            dt = df[base]
            df["Date"] = dt.dt.date
            df["Month"] = dt.dt.to_period("M").astype(str)
            df["DayName"] = dt.dt.day_name()
            df["Hour"] = dt.dt.hour
            break

    if "Time(IST)" in df.columns:
        t = pd.to_datetime(df["Time(IST)"], format="%H:%M:%S", errors="coerce")
        df["Hour"] = t.dt.hour

    return df

def explode_hashtags(df, col="Hashtags"):
    if col not in df:
        return pd.DataFrame()

    t = df[col].fillna("").astype(str).str.replace("\n", " ")
    parts = t.apply(lambda x: re.split(r"[,\s]+", x.strip()))

    tmp = df.copy()
    tmp["__tags__"] = parts
    tmp = tmp.explode("__tags__")
    tmp["Hashtag"] = tmp["__tags__"].str.strip()
    tmp = tmp[tmp["Hashtag"].str.startswith("#") & (tmp["Hashtag"] != "")]
    tmp.drop(columns=["__tags__"], inplace=True)

    return tmp

# ========== FOLLOWERS GAIN ==========
def compute_followers_gain(posts_df, followers_df):

    if "Created date" not in posts_df or "Date" not in followers_df:
        return posts_df

    df = posts_df.copy()
    f = followers_df.copy()

    df["Created date"] = pd.to_datetime(df["Created date"]).dt.date
    f["Date"] = pd.to_datetime(f["Date"]).dt.date

    df = df.sort_values("Created date")
    f = f.sort_values("Date")

    if "Total followers" not in f.columns:
        cols = [
            c for c in ["Sponsored followers", "Organic followers", "Auto-invited followers"]
            if c in f.columns
        ]
        f["Total followers"] = f[cols].sum(axis=1, min_count=1)

    f["Total followers"] = f["Total followers"].cumsum()
    f = f.groupby("Date").tail(1).reset_index(drop=True)

    total = f.groupby("Date")["Total followers"].last().sort_index()

    gained = []

    for i in range(len(df)):
        d = df.iloc[i]["Created date"]
        next_d = df.iloc[i + 1]["Created date"] if i + 1 < len(df) else None

        before = total[total.index <= d]
        after = total[total.index <= next_d] if next_d is not None else total

        if before.empty or after.empty:
            gained.append(0)
        else:
            gained.append(int(after.iloc[-1]) - int(before.iloc[-1]))

    df["Followers gained"] = gained
    return df

# ========== LOAD DATA ==========
content_df = read_excel_safe(CONTENT_PATH)

followers_df = pd.read_excel(FOLLOWERS_PATH, sheet_name="New followers")
followers_df = to_datetime_col(followers_df, "Date")

try:
    xl = pd.ExcelFile(CONTENT_PATH)
    if "All posts" in xl.sheet_names:
        content_df = pd.read_excel(CONTENT_PATH, sheet_name="All posts")
except:
    pass

content_df = derive_time_columns(content_df)
followers_df = to_datetime_col(followers_df, "Date")

for c in ["Impressions", "Clicks", "Likes", "Comments", "Reposts", "Follows", "Engagement rate"]:
    if c in content_df:
        content_df[c] = pd.to_numeric(content_df[c], errors="coerce")

content_df = compute_followers_gain(content_df, followers_df)

# ========== CHATBOT INIT ==========
@st.cache_resource
def init_chatbot_resources():
    out = {"ok": False, "msg": "", "chain": None}

    if FAISS is None:
        out["msg"] = "FAISS missing."
        return out
    if RetrievalQA is None:
        out["msg"] = "LangChain missing."
        return out
    if not VECTOR_DIR.exists():
        out["msg"] = f"Vector DB not found at {VECTOR_DIR}"
        return out

    class LocalEmbeddings(Embeddings):
        def __init__(self):
            self.m = SentenceTransformer("all-MiniLM-L6-v2")
        def embed_documents(self, texts):
            return self.m.encode(texts).tolist()
        def embed_query(self, text):
            return self.m.encode([text])[0].tolist()

    try:
        emb = LocalEmbeddings()
        vs = FAISS.load_local(VECTOR_DIR, embeddings=emb, allow_dangerous_deserialization=True)
    except Exception as e:
        out["msg"] = f"Vector load error: {e}"
        return out

    if not GROQ_KEY:
        out["msg"] = "Groq API key missing."
        return out

    try:
        llm = ChatGroq(groq_api_key=GROQ_KEY, model_name="llama-3.3-70b-versatile")
    except:
        out["msg"] = "LLM init failed."
        return out

    prompt = PromptTemplate.from_template("""
        You are a LinkedIn Strategy Assistant.
        Provide short, clear, data-backed insights.
        Context: {context}
        Question: {question}
    """)

    try:
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vs.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": prompt}
        )
        out["ok"] = True
        out["chain"] = chain
        return out

    except Exception as e:
        out["msg"] = f"Chain error: {e}"
        return out
# =====================================================
# SIDEBAR CHATBOT — CHATGPT STYLE (FINAL & CLEAN)
# =====================================================

st.sidebar.markdown("## 💬 Recommendation Chat Assistant")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.chat-bubble-user {
    background: #DCF8C6;
    padding: 8px 12px;
    border-radius: 16px;
    margin: 6px 0;
    display: block;
}
.chat-bubble-assistant {
    background: #F1F0F0;
    padding: 8px 12px;
    border-radius: 16px;
    margin: 6px 0;
    display: block;
}
</style>
""", unsafe_allow_html=True)

# ---------------- STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []     # list of tuples (role, message)

# ---------------- DISPLAY HISTORY ----------------
def render_history():
    if not st.session_state.chat_history:
        st.sidebar.markdown("<em>No messages yet</em>", unsafe_allow_html=True)
        return
    
    html = ""
    for role, msg in st.session_state.chat_history:
        safe = msg.replace("\n", "<br>")
        if role == "user":
            html += f"<div class='chat-bubble-user'><b>You:</b> {safe}</div>"
        else:
            html += f"<div class='chat-bubble-assistant'><b>Assistant:</b> {safe}</div>"

    st.sidebar.markdown(html, unsafe_allow_html=True)

render_history()

# ---------------- INPUT ----------------
user_msg = st.sidebar.text_input("Ask something:", key="sidebar_input", value="")

if st.sidebar.button("Send"):
    if user_msg.strip():

        # 1️⃣ append user message
        st.session_state.chat_history.append(("user", user_msg.strip()))

        # 2️⃣ call model
        bot = init_chatbot_resources()

        if not bot["ok"]:
            st.sidebar.warning(bot["msg"])
        else:
            try:
                hist_payload = [{"role": r, "text": t} for r, t in st.session_state.chat_history]
                result = bot["chain"]({"query": user_msg, "history": hist_payload})

                response = (
                    result.get("result")
                    or result.get("answer")
                    or result.get("output_text")
                    or str(result)
                )

                # 3️⃣ append assistant response
                st.session_state.chat_history.append(("assistant", response))

            except Exception as e:
                st.session_state.chat_history.append(("assistant", f"Error: {e}"))

        # 4️⃣ refresh to show new messages
        st.rerun()

# ---------------- CLEAR BUTTON ----------------
if st.sidebar.button("Clear Chat"):
    st.session_state.chat_history = []
    st.sidebar.success("Chat cleared!")
    st.rerun()

# =====================================================
# FILTERS SECTION
# =====================================================

with st.expander("🔍 Filters", expanded=False):

    ctype = st.multiselect(
        "Content Type",
        sorted(content_df["Content Type"].dropna().unique())
        if "Content Type" in content_df else []
    )

    ptype = st.multiselect(
        "Post type",
        sorted(content_df["Post type"].dropna().unique())
        if "Post type" in content_df else []
    )

    day_filter = st.multiselect(
        "Day of Week",
        sorted(content_df["DayName"].dropna().unique())
        if "DayName" in content_df else []
    )

    if "Date" in content_df:
        dmin = pd.to_datetime(content_df["Date"]).min()
        dmax = pd.to_datetime(content_df["Date"]).max()
        date_range = st.date_input("Date Range", value=[dmin, dmax])
    else:
        date_range = []


# Apply filters
df = content_df.copy()

if ctype:
    df = df[df["Content Type"].isin(ctype)]
if ptype:
    df = df[df["Post type"].isin(ptype)]
if day_filter:
    df = df[df["DayName"].isin(day_filter)]

if len(date_range) == 2:
    s, e = date_range
    df = df[
        (pd.to_datetime(df["Date"]) >= pd.to_datetime(s)) &
        (pd.to_datetime(df["Date"]) <= pd.to_datetime(e))
    ]


# =====================================================
# KPI SECTION
# =====================================================

st.title("📊 Engagement Intelligence Suite")

df["Total Engagement"] = df[
    ["Likes", "Comments", "Reposts", "Clicks", "Follows"]
].sum(axis=1, skipna=True)

avg_hashtags = df["Hashtags"].dropna().apply(
    lambda x: len(re.findall(r"#[A-Za-z0-9_]+", str(x)))
).mean()

current_er = df["Engagement rate"].mean() * 100 if "Engagement rate" in df else 0
target_er = 17.93 * 1.20

if "Followers gained" in df and df["Followers gained"].notna().any() and df["Followers gained"].sum() != 0:
    avg_followers = df["Followers gained"].mean()
    avg_followers_display = f"{avg_followers:.1f}"
else:
    avg_followers_display = "—"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Posts", df.shape[0])
c2.metric("Total Engagement", f"{int(df['Total Engagement'].sum()):,}")
c3.metric("Avg Engagement Rate", f"{current_er:.2f}%")
c4.metric("Avg Hashtags/Post", f"{avg_hashtags:.1f}")
c5.metric("Avg Followers/Post", avg_followers_display)

st.markdown("### 🎯 Engagement Rate Progress")

cx, cy = st.columns(2)
cx.metric("Current ER", f"{current_er:.2f}%")
cy.metric("Target ER (+20%)", f"{target_er:.2f}%")

er_df = pd.DataFrame({
    "Metric": ["Current ER", "Target ER"],
    "Value": [current_er, target_er]
})

st.plotly_chart(
    px.bar(
        er_df,
        x="Metric",
        y="Value",
        text="Value",
        color="Metric",
        color_discrete_sequence=["#2980b9", "#27ae60"],
        title="Current vs Target ER"
    ),
    use_container_width=True
)
# =====================================================
# TABS SECTION
# =====================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Content Analysis",
    "Timing Analysis",
    "Audience Insights",
    "Post Explorer",
    "Summary"
])


# -----------------------------------------------------
# OVERVIEW TAB
# -----------------------------------------------------
with tab1:
    st.subheader("📅 Monthly Engagement Trends")

    if "Month" in df:

        monthly = df.groupby("Month").agg(
            Avg_ER=("Engagement rate", "mean"),
            Total_Eng=("Total Engagement", "sum")
        ).reset_index()

        # Convert YYYY-MM → Jan 2025
        monthly["Month_Label"] = (
            pd.to_datetime(monthly["Month"].astype(str), format="%Y-%m")
            .dt.strftime("%b %Y")
        )

        # ---- Line Chart: Avg ER ----
        fig1 = px.line(
            monthly,
            x="Month",
            y="Avg_ER",
            markers=True,
            title="Average Engagement Rate by Month"
        )
        fig1.update_xaxes(
            tickmode="array",
            tickvals=monthly["Month"],
            ticktext=monthly["Month_Label"]
        )

        st.plotly_chart(fig1, use_container_width=True, key="overview_line_er")

        # ---- Bar Chart: Total Engagement ----
        fig2 = px.bar(
            monthly,
            x="Month",
            y="Total_Eng",
            title="Total Engagement by Month"
        )
        fig2.update_xaxes(
            tickmode="array",
            tickvals=monthly["Month"],
            ticktext=monthly["Month_Label"]
        )

        st.plotly_chart(fig2, use_container_width=True, key="overview_bar_eng")


# -----------------------------------------------------
# CONTENT ANALYSIS TAB
# -----------------------------------------------------
with tab2:
    st.subheader("📂 Content Type & Hashtag Performance")

    # 1️⃣ Content-Type ER
    if "Content Type" in df:
        ct_perf = df.groupby("Content Type")["Engagement rate"].mean().reset_index()

        st.plotly_chart(
            px.bar(
                ct_perf,
                x="Content Type",
                y="Engagement rate",
                title="Avg Engagement Rate by Content Type"
            ),
            use_container_width=True,
            key="content_type_er"
        )

    # 2️⃣ Hashtag Performance
    tags = explode_hashtags(df)

    if not tags.empty:
        tag_ct = tags.groupby(["Content Type", "Hashtag"])["Engagement rate"].mean().reset_index()

        st.plotly_chart(
            px.bar(
                tag_ct,
                x="Hashtag",
                y="Engagement rate",
                color="Content Type",
                title="Hashtag Performance by Content Type"
            ),
            use_container_width=True,
            key="hashtag_performance"
        )


# -----------------------------------------------------
# TIMING ANALYSIS TAB
# -----------------------------------------------------
with tab3:
    st.subheader("🕒 Time-Based Engagement Insights")

    # 1️⃣ ER by Day of Week
    if "DayName" in df:
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_er = df.groupby("DayName")["Engagement rate"].mean().reindex(order).reset_index()

        st.plotly_chart(
            px.bar(
                day_er,
                x="DayName",
                y="Engagement rate",
                title="Avg ER by Day of Week"
            ),
            use_container_width=True,
            key="timing_day_er"
        )

    # 2️⃣ ER by Posting Hour
    if "Hour" in df:
        df["HourLabel"] = df["Hour"].apply(
            lambda x: f"{int(x % 12 or 12)} {'AM' if x < 12 else 'PM'}"
        )
        hour_er = df.groupby("HourLabel")["Engagement rate"].mean().reset_index()

        st.plotly_chart(
            px.bar(
                hour_er,
                x="HourLabel",
                y="Engagement rate",
                title="Avg ER by Posting Hour (IST)"
            ),
            use_container_width=True,
            key="timing_hour_er"
        )


# -----------------------------------------------------
# AUDIENCE INSIGHTS TAB
# -----------------------------------------------------
with tab4:
    st.subheader("👥 Follower Growth & Audience Insights")

    # 1️⃣ Followers gained per post
    if "Followers gained" in df:

        fig = px.bar(
            df,
            x="Created date",
            y="Followers gained",
            title="Followers Gained per Post"
        )

        df["Month_Label"] = pd.to_datetime(df["Created date"]).dt.strftime("%b %Y")

        unique_months = df.drop_duplicates("Month_Label")[["Created date", "Month_Label"]]

        fig.update_xaxes(
            tickmode="array",
            tickvals=unique_months["Created date"],
            ticktext=unique_months["Month_Label"],
            tickangle=-45
        )

        st.plotly_chart(fig, use_container_width=True, key="audience_followers_per_post")

    # 2️⃣ Followers Growth per Month
    try:
        followers_df["Date"] = pd.to_datetime(followers_df["Date"], errors="coerce")
        followers_df["Month_Period"] = followers_df["Date"].dt.to_period("M")

        monthly_growth = followers_df.groupby("Month_Period")["Total followers"].sum()

        full_range = pd.period_range(
            start=monthly_growth.index.min(),
            end=monthly_growth.index.max(),
            freq="M"
        )
        monthly_growth = monthly_growth.reindex(full_range, fill_value=0)

        growth_df = monthly_growth.reset_index()
        growth_df.columns = ["Month", "Followers Growth"]
        growth_df["Month"] = growth_df["Month"].dt.strftime("%b %Y")

        fig_growth = px.bar(
            growth_df,
            x="Month",
            y="Followers Growth",
            title="Followers Growth per Month",
            text="Followers Growth"
        )
        fig_growth.update_xaxes(tickangle=-45)

        st.plotly_chart(fig_growth, use_container_width=True, key="audience_growth_per_month")

    except Exception as e:
        st.info(f"Followers dataset cannot be processed. ({e})")


# -----------------------------------------------------
# POST EXPLORER TAB
# -----------------------------------------------------
with tab5:
    st.subheader("📑 Post Explorer")

    columns = [
        "Post title",
        "Content Type",
        "Engagement rate",
        "Likes",
        "Comments",
        "Reposts",
        "Follows",
        "Hashtags"
    ]

    cols_available = [c for c in columns if c in df.columns]

    st.dataframe(df[cols_available], use_container_width=True, key="post_explorer_table")

    st.download_button(
        "Download Filtered Posts (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        "filtered_posts.csv",
        "text/csv"
    )


# -----------------------------------------------------
# SUMMARY TAB
# -----------------------------------------------------
with tab6:
    st.subheader("🧠 Summary Insights")

    try: best_ct = df.groupby("Content Type")["Engagement rate"].mean().idxmax()
    except: best_ct = "N/A"

    try: best_day = df.groupby("DayName")["Engagement rate"].mean().idxmax()
    except: best_day = "N/A"

    try: best_hour = df.groupby("Hour")["Engagement rate"].mean().idxmax()
    except: best_hour = "N/A"

    tags = explode_hashtags(df)
    top_tag = tags["Hashtag"].value_counts().idxmax() if not tags.empty else "N/A"

    st.markdown(f"""
    ### 📌 Summary  
    - ⭐ **Best Content Type:** `{best_ct}`  
    - 📅 **Best Day to Post:** `{best_day}`  
    - ⏰ **Best Hour:** `{best_hour}:00`  
    - 🏷️ **Top Hashtag:** `{top_tag}`  
    - 👥 **Avg Followers/Post:** `{avg_followers_display}`  
    """)
