"""
Qwen-only LangGraph AML/Fraud Detection Demo.

Version 1 keeps transaction monitoring, supervision, and risk scoring
deterministic while using Alibaba Cloud Bailian/DashScope native Qwen APIs for:
- Behavioral Analysis Agent
- SAR Reporting Agent
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from qwen_client import QwenAPIError, QwenClient, QwenConfigurationError


class AMLState(TypedDict, total=False):
    transaction: Dict[str, Any]
    monitor_findings: List[str]
    behavior_signals: List[str]
    behavior_findings: str
    risk_score: int
    sar_report: str
    next: str
    trace: List[str]


def _add_trace(state: AMLState, message: str) -> List[str]:
    return state.get("trace", []) + [message]


def _format_items(items: List[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def sample_initial_state() -> AMLState:
    return {
        "transaction": {
            "customer_id": "CUST-001",
            "transaction_id": "TX-98765",
            "amount": 25_000,
            "country": "IR",
            "is_new_counterparty": True,
            "customer_avg_amount": 3_000,
            "transaction_hour": 2,
            "sudden_velocity_increase": True,
            "previous_alerts": 4,
        },
        "trace": [],
    }


def supervisor_agent(state: AMLState) -> AMLState:
    if "monitor_findings" not in state:
        next_agent = "monitoring_agent"
    elif "behavior_findings" not in state:
        next_agent = "behavior_agent"
    elif "risk_score" not in state:
        next_agent = "risk_agent"
    elif state["risk_score"] >= 70 and "sar_report" not in state:
        next_agent = "sar_agent"
    else:
        next_agent = "END"

    return {
        "next": next_agent,
        "trace": _add_trace(state, f"Supervisor routed to: {next_agent}"),
    }


def monitoring_agent(state: AMLState) -> AMLState:
    tx = state["transaction"]
    findings: List[str] = []

    if tx.get("amount", 0) >= 10_000:
        findings.append(f"Large transaction amount >= 10,000 (actual: {tx['amount']})")

    if tx.get("country") in {"IR", "KP", "SY", "CU", "SD"}:
        findings.append(f"High-risk country: {tx['country']}")

    if tx.get("is_new_counterparty"):
        findings.append("New counterparty detected")

    if not findings:
        findings.append("No rule-based red flags detected")

    return {
        "monitor_findings": findings,
        "trace": _add_trace(state, f"Transaction monitoring completed: {findings}"),
    }


def _behavior_signals(tx: Dict[str, Any]) -> List[str]:
    signals: List[str] = []

    avg_amount = tx.get("customer_avg_amount", 1_000)
    if tx.get("amount", 0) > avg_amount * 5:
        signals.append(
            f"Transaction amount is >5x customer average ({tx['amount']} vs avg {avg_amount})"
        )

    hour = tx.get("transaction_hour")
    if hour is not None and 0 <= hour < 5:
        signals.append(f"Unusual transaction hour: {hour:02d}:xx (midnight window)")

    if tx.get("sudden_velocity_increase"):
        signals.append("Sudden increase in transaction velocity detected")

    if not signals:
        signals.append("No major behavioral anomalies detected")

    return signals


def make_behavior_agent(qwen_client: QwenClient) -> Callable[[AMLState], AMLState]:
    def behavior_agent(state: AMLState) -> AMLState:
        tx = state["transaction"]
        signals = _behavior_signals(tx)

        summary = qwen_client.generate(
            system_prompt=(
                "You are an AML behavioral analysis assistant. Use only the "
                "supplied facts. Keep the summary concise, auditable, and avoid "
                "unsupported conclusions."
            ),
            user_prompt=(
                "Summarize the behavioral AML risk in 2-4 concise bullet points.\n"
                "If the transaction appears normal, state that directly.\n\n"
                f"Customer ID: {tx.get('customer_id', 'N/A')}\n"
                f"Transaction ID: {tx.get('transaction_id', 'N/A')}\n"
                f"Amount: {tx.get('amount', 'N/A')}\n"
                f"Country: {tx.get('country', 'N/A')}\n"
                f"Historical average transaction amount: {tx.get('customer_avg_amount', 'N/A')}\n"
                f"Transaction hour: {tx.get('transaction_hour', 'N/A')}\n"
                f"Previous alerts: {tx.get('previous_alerts', 0)}\n"
                "Deterministic behavioral signals:\n"
                f"{_format_items(signals)}"
            ),
            temperature=0.1,
        )

        return {
            "behavior_signals": signals,
            "behavior_findings": summary,
            "trace": _add_trace(state, "Behavioral analysis completed with Qwen"),
        }

    return behavior_agent


def risk_agent(state: AMLState) -> AMLState:
    score = 0

    for finding in state.get("monitor_findings", []):
        if "Large transaction" in finding:
            score += 25
        if "High-risk country" in finding:
            score += 35
        if "New counterparty" in finding:
            score += 15

    for signal in state.get("behavior_signals", []):
        if ">5x customer average" in signal:
            score += 25
        if "Unusual transaction hour" in signal:
            score += 10
        if "velocity" in signal:
            score += 20

    if state.get("transaction", {}).get("previous_alerts", 0) >= 3:
        score += 20

    score = min(score, 100)

    return {
        "risk_score": score,
        "trace": _add_trace(state, f"Risk scoring completed: score={score}"),
    }


def make_sar_agent(qwen_client: QwenClient) -> Callable[[AMLState], AMLState]:
    def sar_agent(state: AMLState) -> AMLState:
        tx = state["transaction"]
        draft = qwen_client.generate(
            system_prompt=(
                "You draft AML suspicious activity reports for internal analyst review. "
                "Use only the provided evidence, keep the language concise, and make "
                "it explicit that this is a draft requiring human review before any "
                "regulatory submission."
            ),
            user_prompt=(
                "Draft a concise SAR with the sections:\n"
                "1. Alert Summary\n"
                "2. Key Transaction Facts\n"
                "3. Behavioral Analysis\n"
                "4. Recommended Next Steps\n\n"
                "Requirements:\n"
                "- This is a draft for human AML analyst review only.\n"
                "- Do not claim any regulatory filing has happened.\n"
                "- Do not add facts that are not listed below.\n\n"
                f"Customer ID: {tx.get('customer_id', 'N/A')}\n"
                f"Transaction ID: {tx.get('transaction_id', 'N/A')}\n"
                f"Amount: {tx.get('amount', 'N/A')}\n"
                f"Country: {tx.get('country', 'N/A')}\n"
                f"Risk Score: {state.get('risk_score', 'N/A')} / 100\n"
                "Transaction monitoring findings:\n"
                f"{_format_items(state.get('monitor_findings', []))}\n"
                "Deterministic behavioral signals:\n"
                f"{_format_items(state.get('behavior_signals', []))}\n"
                "Qwen behavioral analysis:\n"
                f"{state.get('behavior_findings', 'N/A')}"
            ),
            temperature=0.2,
        )

        report = """\
======================================================
  SUSPICIOUS ACTIVITY REPORT — DRAFT (NOT SUBMITTED)
======================================================
** FOR HUMAN AML ANALYST REVIEW ONLY **
** Do NOT submit to regulators without manual review **
------------------------------------------------------
""" + draft

        return {
            "sar_report": report,
            "trace": _add_trace(state, "SAR draft generated with Qwen (pending human review)"),
        }

    return sar_agent


def route_next(state: AMLState) -> str:
    return state["next"]


def build_graph(qwen_client: QwenClient) -> CompiledStateGraph:
    graph = StateGraph(AMLState)

    graph.add_node("supervisor_agent", supervisor_agent)
    graph.add_node("monitoring_agent", monitoring_agent)
    graph.add_node("behavior_agent", make_behavior_agent(qwen_client))
    graph.add_node("risk_agent", risk_agent)
    graph.add_node("sar_agent", make_sar_agent(qwen_client))

    graph.set_entry_point("supervisor_agent")
    graph.add_conditional_edges(
        "supervisor_agent",
        route_next,
        {
            "monitoring_agent": "monitoring_agent",
            "behavior_agent": "behavior_agent",
            "risk_agent": "risk_agent",
            "sar_agent": "sar_agent",
            "END": END,
        },
    )
    graph.add_edge("monitoring_agent", "supervisor_agent")
    graph.add_edge("behavior_agent", "supervisor_agent")
    graph.add_edge("risk_agent", "supervisor_agent")
    graph.add_edge("sar_agent", "supervisor_agent")

    return graph.compile()


def run_demo() -> None:
    try:
        qwen_client = QwenClient()
        app = build_graph(qwen_client)
        result = app.invoke(sample_initial_state())
    except QwenConfigurationError as exc:
        raise SystemExit(
            "Configuration error: "
            f"{exc}\nCopy .env.example to .env and export DASHSCOPE_API_KEY before running."
        ) from exc
    except QwenAPIError as exc:
        raise SystemExit(f"Qwen runtime error: {exc}") from exc

    print("\n=== Execution Trace ===")
    for step in result.get("trace", []):
        print(" ", step)

    print("\n=== Risk Score ===")
    print(f"  {result.get('risk_score', 'N/A')} / 100")

    print("\n=== Behavioral Findings ===")
    print(result.get("behavior_findings", "No behavioral findings generated"))

    print("\n=== SAR Draft ===")
    print(result.get("sar_report", "No SAR generated (risk score below threshold)"))


if __name__ == "__main__":
    run_demo()
