
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


st.set_page_config(
    page_title="House Price predication",
    page_icon="🏡",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7ff, #edf2ff);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

.hero {
    background: linear-gradient(120deg, #172554, #4338ca, #7c3aed);
    padding: 32px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(49, 46, 129, 0.20);
}

.hero h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 10px;
}

.hero p {
    color: #e0e7ff;
    font-size: 16px;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e0e7ff;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 5px 15px rgba(30, 41, 59, 0.06);
}

div[data-testid="stForm"],
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border-radius: 16px;
}

.stButton > button {
    background: linear-gradient(90deg, #4338ca, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 20px;
    font-weight: bold;
    transition: 0.2s;
}

.stButton > button:hover {
    color: white;
    border: 1px solid #4338ca;
    transform: translateY(-2px);
}

.section-title {
    color: #1e1b4b;
    font-size: 24px;
    font-weight: 700;
    margin-top: 15px;
}

div[data-testid="stSidebar"] {
    background: #eef2ff;
}

.footer {
    text-align: center;
    padding: 18px;
    color: #64748b;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <h1>🏡 House Price prediction</h1>
    <p>Smart House Price Prediction Using Machine Learning</p>
    <p>Estimate a house price using area, bedrooms and bathrooms.</p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Demo project: predictions use illustrative sample data, "
    "not actual property-market prices."
)

data = {
    "Area": [
        600, 800, 900, 1000, 1100,
        1200, 1300, 1400, 1500, 1600,
        1700, 1800, 1900, 2000, 2200,
        2400, 2600, 2800, 3000, 3200
    ],
    "Bedrooms": [
        1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
        3, 3, 4, 4, 4, 4, 4, 5, 5, 5
    ],
    "Bathrooms": [
        1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 4, 4, 4, 4
    ],
    "Price": [
        1800000, 2200000, 2600000, 3000000,
        3300000, 3600000, 3900000, 4200000,
        4500000, 4800000, 5200000, 5600000,
        6000000, 6400000, 7000000, 7600000,
        8200000, 9000000, 9800000, 10500000
    ]
}

df = pd.DataFrame(data)


X = df[["Area", "Bedrooms", "Bathrooms"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)


c1, c2, c3 = st.columns(3)

with c1:
    st.metric("🏠 Total Sample Houses", len(df))

with c2:
    st.metric("📐 Average Area", f"{df['Area'].mean():,.0f} sq ft")

with c3:
    st.metric("💰 Average Sample Price",
              f"₹{df['Price'].mean() / 100000:.1f} Lakh")

st.markdown(
    '<div class="section-title">✨ Predict Your House Price</div>',
    unsafe_allow_html=True
)

with st.container(border=True):
    left, right = st.columns([1, 1])

    with left:
        st.subheader("🏘️ Property Details")

        area = st.number_input(
            "House Area (square feet)",
            min_value=200,
            max_value=10000,
            value=1200,
            step=50
        )

        bedrooms = st.selectbox(
            "Number of Bedrooms",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            index=1
        )

        bathrooms = st.selectbox(
            "Number of Bathrooms",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            index=1
        )

        predict = st.button(
            "🔮 Predict House Price",
            use_container_width=True
        )

    with right:
        st.subheader("📋 Selected Property")
        st.write(f"**Area:** {area:,} sq ft")
        st.write(f"**Bedrooms:** {bedrooms}")
        st.write(f"**Bathrooms:** {bathrooms}")
        st.write("**Model:** Linear Regression")
        st.write("**Prediction type:** Estimated price")

if predict:
    new_house = pd.DataFrame({
        "Area": [area],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms]
    })

    price = max(0, float(model.predict(new_house)[0]))

    st.markdown("---")
    st.subheader("🎉 Prediction Result")

    p1, p2 = st.columns(2)

    with p1:
        st.metric(
            "Estimated House Price",
            f"₹{price:,.0f}"
        )

    with p2:
        st.metric(
            "Price in Lakhs",
            f"₹{price / 100000:.2f} Lakh"
        )

    st.caption(
        "This result is based on the sample training data "
        "and should not be used as an actual property valuation."
    )

st.markdown("---")
st.markdown(
    '<div class="section-title">📊 Data Analysis</div>',
    unsafe_allow_html=True
)

chart1, chart2 = st.columns(2)

with chart1:
    st.subheader("Area vs House Price")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(
        df["Area"],
        df["Price"] / 100000,
        s=65,
        alpha=0.8
    )
    ax.set_xlabel("Area (Square Feet)")
    ax.set_ylabel("Price (₹ Lakhs)")
    ax.set_title("House Area and Price")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with chart2:
    st.subheader("Average Price by Bedrooms")

    bedroom_prices = df.groupby("Bedrooms")["Price"].mean() / 100000

    fig, ax = plt.subplots(figsize=(7, 4))
    bedroom_prices.plot(kind="bar", ax=ax)
    ax.set_xlabel("Number of Bedrooms")
    ax.set_ylabel("Average Price (₹ Lakhs)")
    ax.set_title("Bedroom-wise Average Price")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")
st.markdown(
    '<div class="section-title">🧠 Machine Learning Performance</div>',
    unsafe_allow_html=True
)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

e1, e2 = st.columns(2)

with e1:
    st.metric("Mean Absolute Error", f"₹{mae:,.0f}")

with e2:
    st.metric("R² Score", f"{r2:.3f}")

st.caption(
    "The evaluation uses a small illustrative dataset. "
    "Its scores do not represent performance on real property data."
)


with st.expander("📁 View Sample Dataset"):
    st.dataframe(df, use_container_width=True)

st.markdown("""
<div class="footer">
    House Price AI | Python • Scikit-learn • Streamlit • CSS
    <br>
    College Mini Project | Machine Learning
</div>
""", unsafe_allow_html=True)
