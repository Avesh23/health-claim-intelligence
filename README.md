# 🏥 Health-Claim Document Intelligence

An enterprise-grade document processing suite that leverages **Google Gemini 2.0 Flash** to classify medical documents and extract structured data with high precision.

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev/)

## ✨ Key Features

- **Multi-Page Intelligent Processing**: Unlike basic OCR, this system analyzes multi-page PDFs as a single semantic unit.
- **AI-Powered Classification**: Automatically categorizes documents into 9 specific insurance types (Claim Forms, Discharge Summaries, ID Proofs, etc.).
- **Smart Data Extraction**:
  - Extracts 15+ field types including Member IDs, Policy Numbers, and Dates.
  - Handles complex **Line Items** and **Billing Tables** across multiple pages.
  - Provides **Confidence Scores** for every extracted field.
- **Built-in UI**: Includes a modern Streamlit dashboard for easy file uploads and visual results.
- **Robust API**: Built with FastAPI, featuring rate limiting, structured logging, and async processing.

---

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python 3.9+)
- **Frontend**: Streamlit
- **AI Engine**: Google Gemini 2.0 Flash (Multimodal)
- **PDF Engine**: PyMuPDF (Fitz)
- **Environment**: Docker Ready

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9 or higher
- A Google Gemini API Key. [Get one here](https://aistudio.google.com/).

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/health-claim.git
cd health-claim

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
LOG_LEVEL=INFO
```

### 4. Running the Application

**Step 1: Start the Backend API**
```bash
python main.py
```
The API will be live at `http://localhost:8000`. Explore the docs at `/docs`.

**Step 2: Start the Frontend UI**
```bash
streamlit run streamlit_app.py
```
Visit `http://localhost:8501` in your browser.

---

## 📡 API Endpoints

### 🔍 Document Classification
`POST /v1/classify`
Upload documents to identify their type.

**Request:**
```bash
curl -X 'POST' \
  'http://localhost:8000/v1/classify' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@my_bill.pdf'
```

### 📑 Data Extraction
`POST /v1/extract`
Extract structured JSON from recognized document types.

---

## 📂 Supported Document Types

| Category | Key Fields Extracted |
| :--- | :--- |
| **Claim Form** | Member ID, Policy Num, Claimed Amount, Signature Status |
| **Discharge Summary** | Admission Date, Discharge Date, Diagnosis |
| **Invoice/Bill** | Bill Date, Grand Total, Detailed Line Items |
| **ID Proof** | Name, ID Number, Type, Date of Birth |
| **Policy Form** | Customer ID, Expiry Date, Policy Status |

---

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.

---
*Created with ❤️ for more efficient insurance workflows.*
