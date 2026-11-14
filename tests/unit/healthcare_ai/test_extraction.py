"""Offline extraction and UI regressions using exclusively synthetic inputs.

Run without optional dependencies or pytest configuration using:
python -m unittest discover -s tests/unit/healthcare_ai -p test_extraction.py -v
"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

APP_PATH = Path(__file__).resolve().parents[3] / "healthcare_ai" / "app.py"
VALID_ENTITIES = {
    "diagnosis": "Osteoarthritis",
    "treatment_history": "Documented Conservative Management",
    "procedure_request": "Total Knee Arthroplasty",
}
SYNTHETIC_NOTE = "Synthetic example: osteoarthritis, therapy, knee arthroplasty."


def load_app():
    """Import the actual app without UI, SDK, or dataframe dependencies."""
    spec = importlib.util.spec_from_file_location("extraction_test_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    modules = {
        "streamlit": MagicMock(),
        "pandas": MagicMock(),
        "pypdf": MagicMock(),
        "openai": MagicMock(),
    }
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class SessionState(dict):
    """Provide the dictionary and attribute access used by Streamlit state."""

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


class ExtractionTests(unittest.TestCase):
    """Exercise provider responses and the real workbench control flow."""

    @classmethod
    def setUpClass(cls):
        cls.app = load_app()

    def response(self, content, finish_reason="stop", refusal=None):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=finish_reason,
                    message=SimpleNamespace(content=content, refusal=refusal),
                )
            ]
        )

    def extract_response(self, response):
        with (
            patch.object(self.app, "HAS_OPENAI", True),
            patch.object(self.app, "OpenAI") as client,
        ):
            client.return_value.chat.completions.create.return_value = response
            return self.app.LLMClinicalExtractor("synthetic-key").extract(
                SYNTHETIC_NOTE
            )

    def invalid_responses(self):
        yield "malformed JSON", self.response("not JSON")
        yield "empty choices", SimpleNamespace(choices=[])
        yield "missing message", SimpleNamespace(choices=[SimpleNamespace()])
        yield "null content", self.response(None)
        yield "non-text content", self.response({"text": "invalid"})
        yield "truncated", self.response(json.dumps(VALID_ENTITIES), "length")
        yield "filtered", self.response(json.dumps(VALID_ENTITIES), "content_filter")
        yield "refused", self.response(json.dumps(VALID_ENTITIES), refusal="refused")
        for value in (None, [], "text", 5, True):
            yield f"non-object {value!r}", self.response(json.dumps(value))
        for field in VALID_ENTITIES:
            missing = dict(VALID_ENTITIES)
            del missing[field]
            yield f"missing {field}", self.response(json.dumps(missing))
            for value in (None, 1, False, [], {}, "", " \n\t"):
                invalid = {**VALID_ENTITIES, field: value}
                yield f"invalid {field}: {value!r}", self.response(json.dumps(invalid))

    def test_successful_provider_extraction(self):
        result = self.extract_response(self.response(json.dumps(VALID_ENTITIES)))
        self.assertEqual(result, VALID_ENTITIES)
        prediction = self.app.AuthorizationRiskModel().predict_risk(result)
        self.assertEqual(prediction["risk_score"], 90)
        bundle = self.app.generate_fhir_bundle(result, prediction, "Synthetic policy")
        self.assertEqual(bundle["resourceType"], "Bundle")

    def test_successful_fallback_extraction(self):
        with patch.object(self.app, "OpenAI") as client:
            result = self.app.LLMClinicalExtractor("").extract(SYNTHETIC_NOTE)
        client.assert_not_called()
        self.assertEqual(result["procedure_request"], "Total Knee Arthroplasty")
        self.assertEqual(
            result["treatment_history"], VALID_ENTITIES["treatment_history"]
        )

    def test_invalid_model_responses_raise_extraction_error(self):
        for label, response in self.invalid_responses():
            with self.subTest(label=label):
                with self.assertRaises(self.app.ExtractionError):
                    self.extract_response(response)

    def test_provider_exception_is_sanitized(self):
        with (
            patch.object(self.app, "HAS_OPENAI", True),
            patch.object(self.app, "OpenAI") as client,
        ):
            client.return_value.chat.completions.create.side_effect = RuntimeError(
                "synthetic secret must not appear"
            )
            with self.assertRaises(self.app.ExtractionError) as raised:
                self.app.LLMClinicalExtractor("synthetic-key").extract(SYNTHETIC_NOTE)
        self.assertNotIn("synthetic secret", str(raised.exception))

    def test_unavailable_provider_does_not_fall_back(self):
        with patch.object(self.app, "HAS_OPENAI", False):
            with self.assertRaises(self.app.ExtractionError):
                self.app.LLMClinicalExtractor("synthetic-key").extract(SYNTHETIC_NOTE)

    def test_selected_provider_without_key_does_not_fall_back(self):
        with self.assertRaises(self.app.ExtractionError):
            self.app.LLMClinicalExtractor("", require_provider=True).extract(
                SYNTHETIC_NOTE
            )

    def workbench(self):
        ui = MagicMock()
        ui.session_state = SessionState()
        ui.sidebar.selectbox.return_value = "Fallback (Rule-Based)"
        ui.tabs.return_value = (MagicMock(), MagicMock())
        ui.radio.return_value = "Manual Text Note"
        ui.text_area.return_value = SYNTHETIC_NOTE
        ui.button.return_value = True
        ui.columns.return_value = (MagicMock(), MagicMock())
        return ui

    def test_each_invalid_response_blocks_workbench_scoring_and_export(self):
        for label, response in self.invalid_responses():
            with self.subTest(label=label):
                self.assert_workbench_blocked(response=response)

    def test_provider_exception_blocks_workbench_scoring_and_export(self):
        self.assert_workbench_blocked(error=RuntimeError("synthetic provider failure"))

    def assert_workbench_blocked(self, response=None, error=None):
        ui = self.workbench()
        ui.sidebar.selectbox.side_effect = ["OpenAI", "synthetic-model"]
        ui.sidebar.text_input.return_value = "synthetic-key"
        with (
            patch.object(self.app, "st", ui),
            patch.object(self.app, "HAS_OPENAI", True),
            patch.object(self.app, "OpenAI") as client,
            patch.object(self.app.ClinicalPolicyRetriever, "search_policies") as search,
            patch.object(self.app.AuthorizationRiskModel, "predict_risk") as score,
            patch.object(self.app, "generate_fhir_bundle") as export,
        ):
            create = client.return_value.chat.completions.create
            create.return_value = response
            create.side_effect = error
            self.app.main()
        search.assert_not_called()
        score.assert_not_called()
        export.assert_not_called()
        ui.download_button.assert_not_called()
        ui.success.assert_not_called()
        ui.error.assert_called_once()
        self.assertIn("Human review required", ui.error.call_args.args[0])
        self.assertEqual(ui.session_state.audit_logs, [])

    def test_successful_workbench_still_scores_and_exports(self):
        ui = self.workbench()
        with (
            patch.object(self.app, "st", ui),
            patch.object(
                self.app,
                "generate_fhir_bundle",
                wraps=self.app.generate_fhir_bundle,
            ) as export,
        ):
            self.app.main()
        export.assert_called_once()
        self.assertEqual(len(ui.session_state.audit_logs), 1)
        ui.metric.assert_called_once()
        self.assertTrue(ui.download_button.called)

    def test_provider_configuration_failures_block_workbench(self):
        for api_key, has_sdk in (("", True), ("synthetic-key", False)):
            with self.subTest(api_key=bool(api_key), has_sdk=has_sdk):
                ui = self.workbench()
                ui.sidebar.selectbox.side_effect = ["OpenAI", "synthetic-model"]
                ui.sidebar.text_input.return_value = api_key
                with (
                    patch.object(self.app, "st", ui),
                    patch.object(self.app, "HAS_OPENAI", has_sdk),
                    patch.object(self.app, "OpenAI") as client,
                    patch.object(
                        self.app.AuthorizationRiskModel, "predict_risk"
                    ) as score,
                    patch.object(self.app, "generate_fhir_bundle") as export,
                ):
                    self.app.main()
                client.assert_not_called()
                score.assert_not_called()
                export.assert_not_called()
                ui.download_button.assert_not_called()
                self.assertIn("Human review required", ui.error.call_args.args[0])

    def test_scoring_and_export_reject_incomplete_entities_directly(self):
        with self.assertRaises(self.app.ExtractionError):
            self.app.AuthorizationRiskModel().predict_risk({})
        with self.assertRaises(self.app.ExtractionError):
            self.app.generate_fhir_bundle({}, {"risk_score": 20}, "Synthetic policy")


if __name__ == "__main__":
    unittest.main()
