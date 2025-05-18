import streamlit as st
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import itertools


from assign_scores import embed_personalities, compute_scores, recommend_products
from models import load_data, init_models, TOP_K, ALPHA_DEFAULT, BETA_DEFAULT
from insights import post_style_insights

# App Configuration
st.set_page_config(page_title="Influencer Type", layout="wide")


# Sidebar Layout
def sidebar_inputs():
    st.sidebar.header("Find out your influencer personality")
    bio = st.sidebar.text_area(
        "📝 Your Bio", "", placeholder="Place your account bio here..."
    )
    posts_text = st.sidebar.text_area(
        "💬 Recent Posts (one per line)", "", placeholder="Enter your recent captions..."
    )
    posts = [p.strip() for p in posts_text.splitlines() if p.strip()]
    top_k = st.sidebar.number_input("Number of personas", 1, 20, TOP_K)
    analyze = st.sidebar.button("Analyze Influencer")
    return bio, posts, top_k, analyze


# App Header
def render_header():
    st.markdown(
        """
    <div style='background-color:#f9f1cd; padding:10px; border-radius:8px;'>
      <h1 style='margin:0; text-align:center; color:black;'>Influencer Personality</h1>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; font-size:18px; color:#000000;'>Discover your social media personality. 🌟</p>",
        unsafe_allow_html=True,
    )


# Display Persona Cards
def display_personas(results, personalities):
    st.subheader("This is what your posts say about you")
    for r in results:
        persona = next(p for p in personalities if p["label"] == r["label"])
        st.markdown(
            f"""
        <div style='background:#f9f9f9; padding:18px; border-radius:12px; margin-bottom:12px;'>
            <h3 style='margin:0; font-size:22px; color:#222;'>{persona['emoji']} {persona['label']}</h3>
            <p style='margin:6px 0; font-size:16px; color:#444;'>{persona['description']}</p>
            <p style='margin:4px 0; font-size:14px; color:#555;'><strong>Traits:</strong> {', '.join(persona['traits'])}</p>
            <p style='margin:4px 0; font-size:14px; color:#555;'><strong>Also known as:</strong> {', '.join(persona['synonyms'])}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


# Display Post Style Insights
def display_post_style(insights):
    col1, col2, col3, col4 = st.columns(4)

    def styled_box(col, icon, value, label, bg_color, border):
        col.markdown(
            f"""
        <div style='text-align: center; background-color: {bg_color}; padding: 16px; border-radius: 12px; border: 1px solid {border};'>
            <div style='font-size: 28px;'>{icon}</div>
            <div style='font-size: 32px; font-weight: 700; color:#333;'>{value}</div>
            <div style='font-size: 15px; font-weight: 600; color: #444;'>{label}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    styled_box(
        col1,
        "📝",
        insights["avg_post_length_words"],
        "Avg words / post",
        "#fff8ec",
        "#f0e1c0",
    )
    styled_box(
        col2,
        "📏",
        insights["avg_post_length_chars"],
        "Avg characters / post",
        "#f0faff",
        "#cceeff",
    )
    styled_box(
        col3,
        "😄",
        f"{insights['emoji_usage_percent']}%",
        "Posts with emojis",
        "#fff0f6",
        "#f6cde1",
    )
    styled_box(
        col4,
        "❗",
        insights["avg_exclamations_per_post"],
        "Exclamations / post",
        "#edf9f0",
        "#bfe8c7",
    )


# Display Recommendation Charts
def display_charts(df_display, df_recs):
    pie_col, bar_col = st.columns(2)

    with pie_col:
        st.subheader("🎯 Personality Match Breakdown")
        fig2, ax2 = plt.subplots(figsize=(2, 2), dpi=100)
        colors = ["#b8d8f4", "#cce3dc", "#f6d6ad", "#f4cccc", "#dcd6f7"]
        ax2.pie(
            df_display["Match Score"],
            labels=df_display["Persona Type"],
            autopct="%1.1f%%",
            startangle=140,
            colors=colors[: len(df_display)],
            wedgeprops={"edgecolor": "white"},
            textprops={"fontsize": 6, "color": "black"},
        )
        ax2.axis("equal")
        st.pyplot(fig2, clear_figure=True)

    with bar_col:
        st.subheader("📊 Best Fit Sponsorships")
        summary = (
            df_recs.groupby("category")
            .agg(count=("product", "count"), avg_score=("score", "mean"))
            .sort_values(by="avg_score", ascending=True)
        )

        color_cycle = list(itertools.islice(itertools.cycle(colors), len(summary)))
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        ax.barh(
            summary.index, summary["avg_score"], color=color_cycle, edgecolor="#cccccc"
        )
        ax.set_xlabel("Avg Score", fontsize=8)
        ax.set_title("Top Product Categories", fontsize=9)
        st.pyplot(fig)


def main():
    render_header()
    bio, posts, top_k, analyze = sidebar_inputs()

    if not analyze:
        st.markdown(
            "<p style='text-align:center; font-size:16px; color:#000000;'>Enter your bio and posts to begin...</p>",
            unsafe_allow_html=True,
        )
        return

    data_dir = Path("./data")
    personalities = load_data(data_dir / "personality_types_knowledge_base.json")

    with (data_dir / "product_catalog.json").open("r") as f:
        product_catalog = json.load(f)

    embed_model, zero_shot = init_models()
    persona_vecs = embed_personalities(personalities, embed_model)

    results = compute_scores(
        bio,
        posts,
        personalities,
        persona_vecs,
        embed_model,
        zero_shot,
        alpha=ALPHA_DEFAULT,
        beta=BETA_DEFAULT,
        top_k=top_k,
    )

    df = pd.DataFrame(results)
    df["Persona"] = df["label"]
    df_display = df[["Persona", "combined_score"]].rename(
        columns={"Persona": "Persona Type", "combined_score": "Match Score"}
    )
    df_display.index += 1

    top_personas = [r["label"] for r in results[:3]]
    product_recs = recommend_products(
        posts, top_personas, product_catalog, embed_model, top_k=5
    )
    df_recs = pd.DataFrame(product_recs)

    tab1, tab2 = st.tabs(["🎈 Personality", "📊 Analytics"])
    with tab1:
        st.subheader("Post Insights")
        display_post_style(post_style_insights(posts))
        st.markdown("<br><br>", unsafe_allow_html=True)
        display_personas(results, personalities)

    with tab2:
        display_charts(df_display, df_recs)
        st.subheader("🧠 Persona Ranking Table")
        st.dataframe(
            df_display.style.format({"Match Score": "{:.3f}"}), use_container_width=True
        )

        st.subheader("🛍️ Product Recommendations Based on the Posts")
        for rec in product_recs:
            st.markdown(
                f"""
            <div style='background:#f0f0f0; padding:10px; border-radius:6px; margin-bottom:8px;'>
                <strong>{rec['product']}</strong> ({rec['category']}) — 
                <span style='color:#555;'>Match Score: {rec['score']:.2f}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
