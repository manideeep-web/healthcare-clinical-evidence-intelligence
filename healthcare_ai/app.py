import streamlit as st
from typing import Dict, Any
import json
import datetime
import pandas as pd

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# --- 1. LLM / Fallback Clinical Entity Extractor ---
class LLMClinicalExtractor:
    def __init__(self, api_key: str, base_url: str = None, model_name: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

    def extract(self, text: str) -> Dict[str, str]:
        if not self.api_key or not HAS_OPENAI:
            text_lower = text.lower()
            entities = {
                "diagnosis": "General Medical Review",
                "treatment_history": "None documented",
                "procedure_request": "Standard Evaluation"
            }
            if "osteoarthritis" in text_lower or "knee" in text_lower:
                entities["diagnosis"] = "Osteoarthritis / Knee Degeneration"
                entities["procedure_request"] = "Total Knee Arthroplasty"
            elif "chest pain" in text_lower or "coronary" in text_lower:
                entities["diagnosis"] = "Coronary Artery Disease"
                entities["procedure_request"] = "Diagnostic Cardiac Catheterization"
            elif "back pain" in text_lower or "spine" in text_lower:
                entities["diagnosis"] = "Lumbar Spinal Stenosis"
                entities["procedure_request"] = "Lumbar Spinal Fusion"
            if "therapy" in text_lower or "conservative" in text_lower:
                entities["treatment_history"] = "Documented Conservative Management"
            return entities

        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            prompt = f"""
            Analyze the following clinical provider note and extract three fields in valid JSON format:
            1. "diagnosis": The primary medical condition or diagnosis.
            2. "treatment_history": Prior conservative treatments or therapies attempted.
            3. "procedure_request": The specific medical procedure or test being requested for authorization.

            Clinical Note:
            {text}

            Return ONLY valid JSON with keys: diagnosis, treatment_history, procedure_request.
            """
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {
                "diagnosis": f"Extraction Error: {str(e)}",
                "treatment_history": "Unknown",
                "procedure_request": "Unknown"
            }

# --- 2. Advanced Multi-Policy RAG Retriever ---
class ClinicalPolicyRetriever:
    def __init__(self):
        self.policy_database = {
            "Osteoarthritis": (
                "Humana Policy Guideline 402.1 (Orthopedics): Total Knee Arthroplasty requires "
                "documented failure of conservative therapy, including at least 6 weeks of physical "
                "therapy, structured weight management, and NSAID usage prior to authorization."
            ),
            "Coronary Artery Disease": (
                "Aetna Clinical Policy Bulletin 0122 (Cardiology): Diagnostic Cardiac Catheterization "
                "is medically necessary for patients presenting with objective evidence of myocardial ischemia."
            ),
            "Lumbar Spinal Stenosis": (
                "UnitedHealthcare Commercial Policy 2024.R4 (Spine): Lumbar MRI or surgical intervention "
                "requires a minimum of 12 weeks of conservative care and progressive neurological deficit."
            )
        }

    def search_policies(self, diagnosis: str) -> str:
        for key, policy_text in self.policy_database.items():
            if key.lower() in diagnosis.lower():
                return policy_text
        return "General Payer Guideline: Standard medical necessity review required."

# --- 3. Authorization Risk Model ---
class AuthorizationRiskModel:
    def predict_risk(self, clinical_entities: Dict[str, str]) -> Dict[str, Any]:
        risk_score = 20  
        history = clinical_entities.get("treatment_history", "").lower()
        procedure = clinical_entities.get("procedure_request", "").lower()
        
        if history != "none documented" and history != "unknown" and len(history) > 3:
            risk_score += 30 
        if "arthroplasty" in procedure or "fusion" in procedure or "catheterization" in procedure:
            risk_score += 40
            
        if risk_score >= 80:
            tier = "High Complexity - Medical Director Review Required"
        elif risk_score >= 50:
            tier = "Medium Complexity - Nurse Review"
        else:
            tier = "Low Complexity - Auto-Approve Eligible"
            
        return {
            "risk_score": risk_score,
            "review_tier": tier
        }

def extract_text_from_pdf(uploaded_file) -> str:
    if not HAS_PYPDF:
        return "Error: pypdf library is not installed."
    try:
        reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text if text.strip() else "Warning: No extractable text found in this PDF."
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# --- 4. HL7 FHIR Exporter Utility ---
def generate_fhir_bundle(entities: dict, prediction: dict, policy: str) -> dict:
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "entry": [
            {
                "resource": {
                    "resourceType": "Claim",
                    "status": "active",
                    "use": "preauthorization",
                    "diagnosis": [{"diagnosisString": entities.get("diagnosis")}],
                    "procedure": [{"procedureString": entities.get("procedure_request")}],
                    "supportingInfo": [{"category": "Treatment History", "code": entities.get("treatment_history")}]
                }
            },
            {
                "resource": {
                    "resourceType": "RiskAssessment",
                    "status": "final",
                    "prediction": [{
                        "outcome": {"text": prediction.get("review_tier")},
                        "probabilityDecimal": prediction.get("risk_score") / 100.0
                    }],
                    "basis": policy
                }
            }
        ]
    }

# --- 5. Streamlit UI Dashboard with Audit Logging ---
def main():
    st.set_page_config(page_title="Clinical Evidence AI", layout="wide")
    st.title("Healthcare Prior Authorization Intelligence")
    st.markdown("Enterprise AI decision support powered by LLM extraction, multi-domain RAG, FHIR interoperability, and audit compliance.")

    # Initialize session state for audit logging
    if "audit_logs" not in st.session_state:
        st.session_state.audit_logs = []

    # Sidebar configuration for LLM
    st.sidebar.header("AI Provider Configuration")
    provider = st.sidebar.selectbox("Select LLM Provider", ["Fallback (Rule-Based)", "OpenAI", "Groq (Fast & Free Tier)"])
    
    api_key = ""
    model_name = "gpt-4o-mini"
    base_url = None

    if provider == "OpenAI":
        api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
        model_name = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
    elif provider == "Groq (Fast & Free Tier)":
        api_key = st.sidebar.text_input("Enter Groq API Key", type="password", help="Get a free key from console.groq.com")
        base_url = "https://api.groq.com/openai/v1"
        model_name = st.sidebar.selectbox("Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])

    nlp = LLMClinicalExtractor(api_key=api_key, base_url=base_url, model_name=model_name)
    rag = ClinicalPolicyRetriever()
    ml = AuthorizationRiskModel()

    # Tabs for Workspace vs Audit Trail
    tab_eval, tab_audit = st.tabs(["📋 Prior Authorization Workbench", "🔒 Enterprise Compliance & Audit Logs"])

    with tab_eval:
        input_method = st.radio("Select Clinical Input Method:", ["Manual Text Note", "Upload Patient Medical Record (PDF/TXT)"], horizontal=True)

        clinical_note = ""

        if input_method == "Manual Text Note":
            default_text = "Patient is a 55yo presenting with severe chest pain and suspected coronary artery disease. Requesting authorization for diagnostic cardiac catheterization."
            clinical_note = st.text_area("Paste EHR Provider Note:", value=default_text, height=150)
        else:
            uploaded_file = st.file_uploader("Upload Medical Chart (PDF or TXT)", type=["pdf", "txt"])
            if uploaded_file is not None:
                if uploaded_file.type == "application/pdf":
                    clinical_note = extract_text_from_pdf(uploaded_file)
                else:
                    clinical_note = uploaded_file.read().decode("utf-8")
                st.text_area("Extracted Document Content Preview:", value=clinical_note, height=150)

        if st.button("Analyze Case"):
            if not clinical_note.strip():
                st.warning("Please provide or upload a clinical note before running analysis.")
            else:
                with st.spinner("Running AI analysis, policy grounding, and logging compliance trail..."):
                    entities = nlp.extract(clinical_note)
                    policy = rag.search_policies(entities["diagnosis"])
                    prediction = ml.predict_risk(entities)
                    
                    # Log event into audit trail
                    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                    audit_entry = {
                        "Timestamp": timestamp,
                        "Diagnosis": entities.get("diagnosis"),
                        "Procedure Requested": entities.get("procedure_request"),
                        "Risk Score": prediction["risk_score"],
                        "Decision Routing": prediction["review_tier"],
                        "Provider Engine": provider
                    }
                    st.session_state.audit_logs.append(audit_entry)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Extracted Clinical Entities")
                        st.json(entities)
                        st.info(f"**Matched Payer Policy (RAG Retrieval):**\n\n{policy}")

                    with col2:
                        st.subheader("AI Decision Support & Routing")
                        score = prediction['risk_score']
                        tier = prediction['review_tier']
                        
                        st.metric(label="Authorization Complexity Score", value=f"{score}/100")
                        
                        if score >= 80:
                            st.error(f"Routing: {tier}")
                        elif score >= 50:
                            st.warning(f"Routing: {tier}")
                        else:
                            st.success(f"Routing: {tier}")
                    
                    st.markdown("---")
                    st.subheader("Standardized Healthcare Interoperability (HL7 FHIR)")
                    fhir_bundle = generate_fhir_bundle(entities, prediction, policy)
                    
                    st.download_button(
                        label="📥 Download Decision as HL7 FHIR Bundle (JSON)",
                        data=json.dumps(fhir_bundle, indent=2),
                        file_name="prior_auth_fhir_bundle.json",
                        mime="application/json"
                    )
                    
                    st.success("Analysis Completed & Logged to Compliance Registry Successfully")

    with tab_audit:
        st.subheader("Session Audit Trail & Compliance Registry")
        st.markdown("Tracks all automated prior authorization evaluations executed during this active session for regulatory reporting.")
        
        if not st.session_state.audit_logs:
            st.info("No cases analyzed yet during this session. Run an analysis in the workbench tab to generate audit records.")
        else:
            df_audit = pd.DataFrame(st.session_state.audit_logs)
            st.dataframe(df_audit, use_container_width=True)
            
            csv_data = df_audit.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Export Compliance Audit Report (CSV)",
                data=csv_data,
                file_name="clinical_ai_audit_report.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()