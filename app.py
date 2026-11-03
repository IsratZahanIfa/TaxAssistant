import streamlit as st

st.set_page_config(page_title="TaxAssistant BD", page_icon="⚖️", layout="wide")

st.title("⚖️ Bangladesh TaxAssistant AI")
st.caption("Neuro-Symbolic Legal & Tax RAG Framework (Income Tax Act 2023)")

tab1, tab2 = st.tabs(["💬 AI Legal Assistant", "🧮 Income Tax Calculator"])

with tab1:
    st.subheader("Query Tax Laws & SROs")
    user_query = st.text_input("Enter your tax question:")
    if st.button("Ask Assistant"):
        if user_query:
            with st.spinner("Retrieving legal clauses..."):
                # Call RAG logic here
                st.markdown("**Answer:** Under Section 32 of Income Tax Act 2023, statutory exemption is allowed up to 1/3rd of salary or Tk 500,000 (lower).")
                st.info("Source: Income Tax Act 2023 - Section 32")

with tab2:
    st.subheader("Tax Calculator (Section 32 Exemption)")
    gross_salary = st.number_input("Annual Gross Salary (BDT)", min_value=0.0, value=600000.0, step=10000.0)
    category = st.selectbox("Taxpayer Category", ["General (3.5L)", "Female / Senior 65+ (4L)", "Disabled (4.75L)", "Freedom Fighter (5L)"])
    
    if st.button("Calculate Tax"):
        statutory_exemption = min(gross_salary / 3, 500000)
        taxable_salary = gross_salary - statutory_exemption
        st.success(f"Statutory Exemption (Sec 32): Tk {statutory_exemption:,.2f}")
        st.metric(label="Net Taxable Salary", value=f"Tk {taxable_salary:,.2f}")

        <div class="container">
    <h2>Streamlit Tax Workstation</h2>
    <iframe src="http://localhost:8501" width="100%" height="750px" style="border:1px solid #e2e8f0; border-radius:8px;"></iframe>
</div>