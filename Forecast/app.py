import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from groq import Groq
from dotenv import load_dotenv
import pdfplumber
from openpyxl import Workbook

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit UI Setup
st.set_page_config(page_title="AI Invoice Coding", page_icon="📄", layout="wide")
st.title("📄 AI-Driven Invoice Coding")
st.subheader("Upload an invoice PDF and a Chart of Accounts (Excel/CSV)")

# File Upload - Invoice PDF
invoice_file = st.file_uploader("Upload Invoice PDF", type=["pdf"])
coa_file = st.file_uploader("Upload Chart of Accounts (Excel/CSV)", type=["xlsx", "csv"])

if invoice_file and coa_file:
    # Extract text from PDF
    with pdfplumber.open(invoice_file) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    
    st.write("### Extracted Invoice Text")
    st.text(text[:1000])  # Show a preview of extracted text
    
    # Load Chart of Accounts
    if coa_file.name.endswith(".csv"):
        coa_df = pd.read_csv(coa_file)
    else:
        coa_df = pd.read_excel(coa_file)
    
    st.write("### Preview of Chart of Accounts", coa_df.head())
    
    # AI-Based Invoice Coding
    st.subheader("🤖 AI-Generated Invoice Coding")
    client = Groq(api_key=GROQ_API_KEY)
    
    prompt = f"""
    You are an expert accountant. Based on the extracted invoice text, map the relevant line items
    to the appropriate GL codes from the Chart of Accounts. Return the results in a structured format
    with columns: 'Description', 'Amount', 'GL Code', 'GL Description'.
    
    Invoice text:
    {text[:2000]}
    
    Chart of Accounts:
    {coa_df.to_json()}
    """
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert accountant specialized in invoice coding."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
        )
        ai_coding = response.choices[0].message.content
        st.write(ai_coding)
    except Exception as e:
        st.error(f"🚨 AI Processing Error: {str(e)}")
    
    # Export to Excel
    st.subheader("📤 Export Coded Invoice")
    output_file = "coded_invoice.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.append(["Description", "Amount", "GL Code", "GL Description"])
    for line in ai_coding.split("\n")[1:]:  # Skip header row
        ws.append(line.split(","))
    wb.save(output_file)
    
    with open(output_file, "rb") as f:
        st.download_button("Download Coded Invoice", f, file_name=output_file, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
