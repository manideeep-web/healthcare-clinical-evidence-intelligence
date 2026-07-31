from typing import Dict


class ClinicalEntityExtractor:
    """
    Wrapper class to interface with OpenMed's clinical NLP engine.
    Currently using regex/heuristics for the prototype phase.
    TODO: Connect to OpenMed's OnnxTokenClassifier for prod deployment.
    """

    def __init__(self):
        self.model_loaded = True

    def extract(self, text: str) -> Dict[str, str]:
        text_lower = text.lower()

        entities = {
            "diagnosis": "Unknown",
            "treatment_history": "None documented",
            "procedure_request": "Not specified",
        }

        if "osteoarthritis" in text_lower:
            entities["diagnosis"] = "Osteoarthritis"

        if "physical therapy" in text_lower or "pt" in text_lower:
            entities["treatment_history"] = "Physical Therapy"

        if "knee replacement" in text_lower or "arthroplasty" in text_lower:
            entities["procedure_request"] = "Total Knee Arthroplasty"

        return entities
