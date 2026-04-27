import streamlit as st
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("HEALTH_CLAIM_API_KEY", "")

st.set_page_config(
    page_title="Health-Claim Document Intelligence",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Health-Claim Document Intelligence")
st.markdown("""
Extract structured data from medical bills, insurance letters, and ID proofs using Gemini 2.0.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("X-API-Key", value=API_KEY, type="password")
    st.info("Ensure the FastAPI backend is running at: " + API_URL)

# File uploader
uploaded_files = st.file_uploader(
    "Upload Documents (PDF, PNG, JPEG, HEIC, WebP, TXT)", 
    type=["pdf", "png", "jpg", "jpeg", "heic", "webp", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    cols = st.columns(2)
    with cols[0]:
        if st.button("🔍 Classify Only"):
            with st.spinner("Classifying documents..."):
                files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
                headers = {"X-API-Key": api_key}
                try:
                    response = requests.post(f"{API_URL}/classify", files=files, headers=headers)
                    if response.status_code == 200:
                        results = response.json().get("results", [])
                        for res in results:
                            st.subheader(f"📄 {res['filename']}")
                            if res.get("error"):
                                st.error(res["error"])
                                continue
                            
                            conf_str = f"{res['confidence']:.2f}" if res.get("confidence") is not None else "N/A"
                            st.info(f"Classification: **{res['category'].upper()}** (Conf: {conf_str})")
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    with cols[1]:
        if st.button("📑 Extract Structured Data"):
            with st.spinner("Processing consolidated extraction..."):
                files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
                headers = {"X-API-Key": api_key}
                try:
                    response = requests.post(f"{API_URL}/extract", files=files, headers=headers)
                    if response.status_code == 200:
                        results = response.json().get("results", [])
                        for res in results:
                            st.subheader(f"📄 {res['filename']}")
                            if res.get("error"):
                                st.error(res["error"])
                                continue
                            
                            st.info(f"Category: **{res['document_category'].upper()}**")
                            
                            data_with_confidence = res.get("data", {})
                            display_data = []
                            for field, content in data_with_confidence.items():
                                if field in ("billing_items", "line_items") and isinstance(content, list):
                                    continue
                                
                                if isinstance(content, dict) and "value" in content:
                                    display_data.append({
                                        "Field": field.replace("_", " ").title(),
                                        "Value": content["value"],
                                        "Confidence": content["confidence"]
                                    })
                            
                            if display_data:
                                st.table(display_data)
                                
                            # Consolidated list items
                            for list_field in ("billing_items", "line_items"):
                                items = data_with_confidence.get(list_field)
                                if items:
                                    st.write(f"**Merged {list_field.replace('_', ' ').title()} (All Pages):**")
                                    flat_items = []
                                    for item in items:
                                        flat_item = {}
                                        for k, v in item.items():
                                            if isinstance(v, dict):
                                                flat_item[k.replace("_", " ").title()] = f"{v['value']} (Conf: {v['confidence']:.2f})"
                                        flat_items.append(flat_item)
                                    st.table(flat_items)
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")
else:
    st.info("Please upload one or more files to begin.")

st.divider()
st.caption("Consolidated Multi-Page Extraction Powered by Gemini 2.0 Flash")
