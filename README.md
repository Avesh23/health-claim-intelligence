# Bill Document Classification & Data Extraction API

A robust FastAPI-based service that leverages Google Gemini to classify documents and extract structured data from medical bills, insurance letters, and more.

## 🚀 Features

- **Multi-File Processing**: Upload multiple documents and images simultaneously.
- **Page-Wise Classification**: PDFs are rendered page by page as images so mixed-document files can return different labels per page.
- **Fixed Labels for Claims Workflows**: Classification is constrained to `claim form`, `discharge summary`, `id proof`, `invoice/bill`, `proposal form`, `policy form`, or `other`.
- **Structured Data Extraction**: Automatically extracts 9 key fields:
  - Member ID, Policy Number, Claim/Treatment Dates, Claimed Amount, Location, Bank Amount, Signature Status, and detailed Line Items.
- **Async Execution**: Processes batch uploads concurrently for high performance.
- **Robust Logging**: Comprehensive logging for requests, API calls, and errors.

---

## 🛠️ Setup

### 1. Prerequisites
- Python 3.9+
- A Google Gemini API Key (get one from [Google AI Studio](https://aistudio.google.com/))

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/kaivalya-2004/health-claim.git
cd health-claim

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_api_key_here
LOG_LEVEL=INFO
```

---

## 🏃 Running the API

Start the development server:
```bash
python main.py
```
The API will be available at `http://localhost:8000`.

- **Interative Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Redoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📡 Endpoints

### 1. Classify Documents
`POST /v1/classify`
- **Description**: Identifies the category and confidence for one or more files.
- **Behavior**: PDFs are rendered into page images and classified page by page. Images and text files return a single page result.
- **Input**: Form-data with one or more `files`.
- **Supported Formats**: PDF, PNG, JPEG, WebP, HEIC, TXT.

**Response Example:**
```json
{
  "results": [
    {
      "filename": "bill.pdf",
      "pages": [
        {
          "page_number": 1,
          "category": "claim form",
          "confidence": 0.97,
          "error": null
        },
        {
          "page_number": 2,
          "category": "id proof",
          "confidence": 0.94,
          "error": null
        },
        {
          "page_number": 3,
          "category": "invoice/bill",
          "confidence": 0.99,
          "error": null
        }
      ],
      "error": null
    }
  ]
}
```

### 2. Extract Data
`POST /v1/extract`
- **Description**: Extracts structured JSON data from one or more files.
- **Input**: Form-data with one or more `files`.

**Response Example:**
```json
{
  "results": [
    {
      "filename": "HC-1.pdf",
      "data": {
        "member_id": "MEM123",
        "policy_number": "POL-789",
        "claim_date": "15-03-2025",
        "treatment_date": "10-03-2025",
        "claimed_amount": "₹45,000",
        "line_items": [
          {"description": "Consultation", "amount": "₹2,000", "quantity": "1"}
        ],
        "signature": "present",
        "location": "Mumbai",
        "bank_amount": "₹45,000"
      }
    }
  ]
}
```

---

## 📂 Project Structure

- `main.py`: Application entry point and middleware configuration.
- `core/`: Logging and exception handler configurations.
- `models/`: Pydantic models for request/response validation.
- `routers/`: API route definitions (v1).
- `services/`: Core logic and integration with Google Generative AI.
- `requirements.txt`: Project dependencies.
