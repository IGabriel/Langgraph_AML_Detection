"""
Deterministic LangGraph AML/Fraud Detection Demo

A self-contained, runnable example of a hierarchical multi-agent workflow
for Anti-Money Laundering (AML) and fraud detection.  No OpenAI or external
API keys are required – every agent node is implemented as a plain Python
function.

Agents
------
- Supervisor Agent        : reads shared state and routes to the next agent.
- Transaction Monitor     : rule-based checks (large amount, high-risk country,
                            new counterparty).
- Behavioral Analysis     : checks deviations from customer norms (amount
                            multiples, unusual hour, velocity spike).
- Risk Scoring Agent      : produces an explainable 0-100 risk score.
- SAR Reporting Agent     : generates a SAR *draft* for human AML analyst
                            review — NOT an automatic regulatory submission.
"""

from typing import Any, Dict, List

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class AMLState(TypedDict, total=False):
    transaction: Dict[str, Any]
    monitor_findings: List[str]
    behavior_findings: List[str]
    risk_score: int
    sar_report: str
    next: str
    trace: List[str]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _add_trace(state: AMLState, message: str) -> List[str]:
    return state.get("trace", []) + [message]


# ---------------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------------

def supervisor_agent(state: AMLState) -> AMLState:
    """Route to the next specialist agent based on what is missing in state."""
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
    """Rule-based Transaction Monitoring: flag large amount, high-risk country,
    and new counterparty."""
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


def behavior_agent(state: AMLState) -> AMLState:
    """Behavioral Analysis: detect deviations from customer average amount,
    unusual transaction hour, and sudden velocity spike."""
    tx = state["transaction"]
    findings: List[str] = []

    avg_amount = tx.get("customer_avg_amount", 1_000)
    if tx.get("amount", 0) > avg_amount * 5:
        findings.append(
            f"Transaction amount is >5x customer average "
            f"({tx['amount']} vs avg {avg_amount})"
        )

    hour = tx.get("transaction_hour")
    if hour is not None and 0 <= hour < 5:
        findings.append(f"Unusual transaction hour: {hour:02d}:xx (midnight window)")

    if tx.get("sudden_velocity_increase"):
        findings.append("Sudden increase in transaction velocity detected")

    if not findings:
        findings.append("No major behavioral anomalies detected")

    return {
        "behavior_findings": findings,
        "trace": _add_trace(state, f"Behavioral analysis completed: {findings}"),
    }


def risk_agent(state: AMLState) -> AMLState:
    """Risk Scoring: produce an explainable 0-100 score from monitoring and
    behavioral findings."""
    score = 0

    for finding in state.get("monitor_findings", []):
        if "Large transaction" in finding:
            score += 25
        if "High-risk country" in finding:
            score += 35
        if "New counterparty" in finding:
            score += 15

    for finding in state.get("behavior_findings", []):
        if ">5x customer average" in finding:
            score += 25
        if "Unusual transaction hour" in finding:
            score += 10
        if "velocity" in finding:
            score += 20

    tx = state.get("transaction", {})
    if tx.get("previous_alerts", 0) >= 3:
        score += 20

    score = min(score, 100)

    return {
        "risk_score": score,
        "trace": _add_trace(state, f"Risk scoring completed: score={score}"),
    }


def sar_agent(state: AMLState) -> AMLState:
    """SAR Reporting: generate a Suspicious Activity Report *draft*.

    IMPORTANT: This report is a draft for human AML analyst review only.
    It must NOT be submitted to regulators without manual review and approval
    by a qualified compliance officer.
    """
    tx = state["transaction"]

    monitor_lines = "\n  - ".join(state.get("monitor_findings", []))
    behavior_lines = "\n  - ".join(state.get("behavior_findings", []))

    report = f"""\
======================================================
  SUSPICIOUS ACTIVITY REPORT — DRAFT (NOT SUBMITTED)
======================================================
** FOR HUMAN AML ANALYST REVIEW ONLY **
** Do NOT submit to regulators without manual review **
------------------------------------------------------
Customer ID     : {tx.get("customer_id", "N/A")}
Transaction ID  : {tx.get("transaction_id", "N/A")}
Amount          : {tx.get("amount", "N/A")}
Country         : {tx.get("country", "N/A")}
Risk Score      : {state.get("risk_score", "N/A")} / 100
------------------------------------------------------

Transaction Monitoring Findings:
  - {monitor_lines}

Behavioral Analysis Findings:
  - {behavior_lines}

Rationale:
  This transaction was flagged due to a combination of rule-based red flags,
  behavioral anomalies, and/or historical customer risk indicators that
  collectively exceed the institution's risk threshold.

Recommended Action:
  Escalate to AML compliance analyst for manual review and determination
  before any regulatory submission.  No automated filing has occurred.
======================================================"""

    return {
        "sar_report": report,
        "trace": _add_trace(state, "SAR draft generated (pending human review)"),
    }


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def route_next(state: AMLState) -> str:
    """Return the name of the next node as decided by the supervisor."""
    return state["next"]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> "CompiledGraph":  # noqa: F821
    graph = StateGraph(AMLState)

    graph.add_node("supervisor_agent", supervisor_agent)
    graph.add_node("monitoring_agent", monitoring_agent)
    graph.add_node("behavior_agent", behavior_agent)
    graph.add_node("risk_agent", risk_agent)
    graph.add_node("sar_agent", sar_agent)

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

    # Each specialist returns control to the supervisor after completing its task.
    graph.add_edge("monitoring_agent", "supervisor_agent")
    graph.add_edge("behavior_agent", "supervisor_agent")
    graph.add_edge("risk_agent", "supervisor_agent")
    graph.add_edge("sar_agent", "supervisor_agent")

    return graph.compile()


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def run_demo() -> None:
    app = build_graph()

    # Sample high-risk transaction
    initial_state: AMLState = {
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

    result = app.invoke(initial_state)

    print("\n=== Execution Trace ===")
    for step in result.get("trace", []):
        print(" ", step)

    print("\n=== Risk Score ===")
    print(f"  {result.get('risk_score', 'N/A')} / 100")

    print("\n=== SAR Report ===")
    print(result.get("sar_report", "No SAR generated (risk score below threshold)"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_demo()
