import os
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import patch

import qwen_aml_demo
from qwen_client import QwenClient, QwenConfigurationError


class FakeResponse:
    def __init__(self, content: str):
        self.status_code = HTTPStatus.OK
        self.code = ""
        self.message = ""
        self.output = SimpleNamespace(
            choices=[{"message": {"content": content}}],
            text=None,
        )


class QwenAMLDemoTests(unittest.TestCase):
    def test_qwen_client_requires_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(QwenConfigurationError):
                QwenClient()

    def test_qwen_graph_runs_with_mocked_dashscope(self) -> None:
        responses = [
            FakeResponse("- Behavioral anomaly summary from Qwen."),
            FakeResponse(
                "1. Alert Summary\n"
                "High-risk transfer requires analyst review.\n"
                "2. Key Transaction Facts\n"
                "Large amount, high-risk geography, and prior alerts.\n"
                "3. Behavioral Analysis\n"
                "Activity is inconsistent with the customer's baseline.\n"
                "4. Recommended Next Steps\n"
                "Escalate for human AML analyst review before any filing."
            ),
        ]

        with patch.dict(
            os.environ,
            {"DASHSCOPE_API_KEY": "test-api-key", "QWEN_MODEL": "qwen-plus"},
            clear=True,
        ):
            with patch("dashscope.Generation.call", side_effect=responses) as mock_call:
                app = qwen_aml_demo.build_graph(QwenClient())
                result = app.invoke(qwen_aml_demo.sample_initial_state())

        self.assertEqual(result["risk_score"], 100)
        self.assertIn("Behavioral anomaly summary from Qwen", result["behavior_findings"])
        self.assertIn("SUSPICIOUS ACTIVITY REPORT — DRAFT", result["sar_report"])
        self.assertIn("FOR HUMAN AML ANALYST REVIEW ONLY", result["sar_report"])
        self.assertEqual(mock_call.call_count, 2)
        self.assertEqual(mock_call.call_args_list[0].kwargs["api_key"], "test-api-key")
        self.assertEqual(mock_call.call_args_list[0].kwargs["model"], "qwen-plus")
        self.assertEqual(mock_call.call_args_list[0].kwargs["result_format"], "message")


if __name__ == "__main__":
    unittest.main()
