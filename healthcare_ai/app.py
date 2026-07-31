import streamlit as st
from typing import Dict, Any

# --- 1. NLP Entity Extractor ---
class ClinicalEntityExtractor:
    def __init__(self):
        self.model_loaded = True 
        
    def extract(self, text: str) -> Dict[str, str]:
        text_lower = text.lower()
        entities = {
            "diagnosis": "Unknown",
            "treatment_health": "None documented",
            "procedure_request": "Not specified"
        }
        if "osteoarthritis" in text_lower:
            entities["diagnosis"] = "Osteoarthritis"
        if "physical therapy" in text_lower or "pt" in text_lower:
            entities["treatment_history"] = "Physical Therapy"
        if "knee replacement" in text_lower or "arthroplasty" in text_lower:
            entities["procedure_request"] = "Total Knee Arthroplasty"
        return entities

# --- 2. RAG Policy Retriever ---
class ClinicalPolicyRetriever:
    def __init__(self):
        self.mock_vector_store = {
            "Osteoarthritis": (
                "Humana Policy Guideline 402.1: Total Knee Arthroplasty requires "
                "documented failure of conservative therapy, including at least "
                "6 weeks of physical therapy and NSAID usage, prior to surgical authorization."
            )
        }

    def search_policies(self, diagnosis: str) -> str:
        if diagnosis in self.mock_vector_store:
            return self.mock_vector_store[diagnosis]
        return "No specific policy found for this diagnosis. Standard medical necessity review required."

# --- 3. Authorization Risk Model ---
class AuthorizationRiskModel:
    def predict_risk(self, clinical_entities: Dict[str, str]) -> Dict[str, Any]:
        risk_score = 15  
        if clinical_entities.get("treatment_history") == "Physical Therapy":
            risk_score += 25 
        if "Arthroplasty" in clinical_entities.get("procedure_request", ""):
            risk_score += 45
            
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

# --- 4. Streamlit UI Dashboard ---
def main():
    st.set_page_config(page_title="Clinical Evidence AI", layout="wide")
    st.title("Healthcare Prior Authorization Intelligence")
    st.markdown("An AI-driven decision support tool for clinical reviews.")

    nlp = ClinicalEntityExtractor()
    rag = ClinicalPolicyRetriever()
    ml = AuthorizationRiskModel()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Clinical Notes Input")
        default_text = "Patient is a 68yo with severe osteoarthritis of the right knee. They have completed 8 weeks of physical therapy without improvement. Requesting authorization for total knee replacement."
        clinical_note = st.text_area("Paste EHR Provider Note:", value=default_text, height=150)
        
        if st.button("Analyze Case"):
            with st.spinner("Processing via OpenMed NLP..."):
                entities = nlp.extract(clinical_note)
                policy = rag.search_policies(entities["diagnosis"])
                prediction = ml.predict_risk(entities)
                
                st.success("Extraction Complete")
                st.json(entities)
                
                st.session_state['policy'] = policy
                st.session_state['prediction'] = prediction

    with col2:
        st.subheader("2. AI Decision Support")
        if 'policy' in st.session_state:
            st.info("**Relevant Medical Policy (RAG Retrieval):**\n\n" + st.session_state['policy'])
            
            st.markdown("### Risk Analysis")
            score = st.session_state['prediction']['risk_score']
            tier = st.session_state['prediction']['review_tier']
            
            st.metric(label="Authorization Complexity Score", value=f"{score}/100")
            
            if score >= 80:
                st.error(f"Routing: {tier}")
            elif score >= 50:
                st.warning(f"Routing: {tier}")
            else:
                st.success(f"Routing: {tier}")

if __name__ == "__main__":
    main()