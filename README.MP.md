# Healthcare Clinical Evidence & Prior Authorization Intelligence

Hey there! Welcome to the repository for my healthcare AI decision-support prototype. 

If you've ever looked at how clinical prior authorizations are handled, you know it's often a painfully manual, slow process bogged down by unstructured notes and complex payer guidelines. This project is a hands-on exploration of how modern AI—specifically combining clinical NLP, Retrieval-Augmented Generation (RAG), and risk-scoring models—can streamline that workflow.

---

## What This App Does

Instead of forcing reviewers to dig through endless documentation, this tool automates the heavy lifting:
* **Flexible Document Intake:** Paste clinical provider notes directly or drag and drop actual PDF medical records.
* **Clinical Entity Extraction:** Automatically parses out key details like diagnoses (e.g., osteoarthritis), treatment histories (e.g., physical therapy), and requested surgical procedures.
* **Policy Grounding (RAG):** Matches patient conditions against clinical guidelines and payer policy requirements.
* **Automated Risk Routing:** Computes a transparent complexity score (0–100) and routes the case into the right review bucket—whether it's auto-approval, nurse review, or medical director escalation.

---

## Check It Out Live
You can test the interactive Streamlit dashboard right in your browser:
👉 **[Open Live App](https://your-streamlit-app-url.streamlit.app)** *(Make sure to update this with your actual deployed URL)*

---

## 📂 Repository Structure
```text
healthcare-clinical-evidence-intelligence/
├── healthcare_ai/
│   ├── app.py                 # The interactive Streamlit dashboard
│   ├── nlp/                   # Clinical NLP extraction modules
│   ├── rag/                   # Policy retrieval and vector search logic
│   └── models/                # Risk scoring and decision routing models
├── openmed/                   # Core clinical framework & grounding tools
└── README.md                  # Project overview