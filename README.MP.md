# Healthcare Prior Authorization Intelligence Platform (OpenMed)

[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-2%2C000+-F5E27A?style=for-the-badge&labelColor=0E1116)](https://huggingface.co/OpenMed)

Ever dealt with the absolute headache of medical prior authorizations? Doctors spend hours digging through dense payer guidelines, and patients wait weeks for approvals. 

I built **OpenMed** to change that. It’s an AI-powered decision support prototype designed to instantly read clinical notes (or full medical PDF charts), check them against real-world insurance policies, predict approval risk, and package everything into industry-standard healthcare formats.

---

## What It Can Do

* ** Drop-in PDF & Text Support:** Paste doctor notes or drag-and-drop actual patient medical records—the app extracts the text automatically.
* ** Smart Clinical Parsing:** Uses a robust local extractor out-of-the-box, with optional plug-and-play support for high-speed LLMs (like OpenAI or Groq) to pull out diagnoses, treatments, and requested procedures.
* ** Multi-Policy RAG:** Cross-references clinical notes against real guidelines from major payers (Humana, Aetna, UnitedHealthcare) across specialties like orthopedics, cardiology, and spine care.
* ** Intelligent Risk Scoring:** Analyzes conservative treatment timelines to score the case complexity and route it appropriately (*Auto-Approve*, *Nurse Review*, or *Medical Director Review*).
* ** HL7 FHIR Interoperability:** Instantly packages the decision into a standard HL7 FHIR JSON bundle (`Claim` and `RiskAssessment` resources) for seamless EHR integration.
* ** Compliance & Audit Logging:** Tracks every evaluation made during your session and lets compliance officers export a full CSV audit report.
* ** Container Ready:** Fully packaged with a clean Dockerfile so it can be spun up anywhere in seconds.

---

## Tech Stack
* **Python 3.10+**
* **Streamlit** (for a fast, clean UI dashboard)
* **PyPDF** (for document text extraction)
* **OpenAI SDK / Groq API** (for generative LLM extraction)
* **Pandas & JSON** (for data handling and audit trails)
* **HL7 FHIR & Docker**

---

## Getting Started Locally

Want to run it on your own machine? It takes less than a minute:

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
   cd YOUR_REPOSITORY