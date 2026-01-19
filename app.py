import os
import warnings
import logging
from dotenv import load_dotenv

import gradio as gr

# AutoGen
from autogen import ConversableAgent, GroupChat, GroupChatManager

# --- Hygiene ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("autogen.oai.client").setLevel(logging.ERROR)

load_dotenv()

# --- Config ---
MODEL_NAME = "gpt-4o-mini"
MAX_ROUND = 5

# IMPORTANT:
# For many AutoGen setups, you do NOT need to pass api_key in llm_config if OPENAI_API_KEY is set.
llm_config = {
    "config_list": [
        {"model": MODEL_NAME}
    ],
    # Optional knobs (safe defaults)
    "temperature": 0.2,
}

# Disable docker execution (matches your notebook)
code_execution_config = {"use_docker": False}

from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class ConsultationOutput(BaseModel):
    """Structured, validated output for a healthcare triage consultation.

    Defines the machine-readable schema for the final consultation
    response produced by the consultation agent. This model enforces
    consistent, safety-aware outputs that can be validated, rendered
    in the UI, or integrated into downstream systems.

    The fields are intentionally non-diagnostic and designed for
    educational triage and decision support only.

    Attributes:
        urgency_level: Overall urgency assessment ("none", "low",
            "medium", or "high").
        possible_conditions: Non-diagnostic symptom-based possibilities.
        self_care: Conservative home care or OTC guidance.
        see_doctor_if: Conditions under which professional medical care
            should be sought.
        emergency_now_if: Warning signs requiring immediate medical
            attention.
        clarifying_questions: Follow-up questions to gather missing
            information.
        summary: User-facing summary of recommended next steps.
    """
    urgency_level: Literal["none", "low", "medium", "high"] = Field(
        ..., description="Overall urgency of the situation."
    )
    possible_conditions: List[str] = Field(
        default_factory=list, description="Non-diagnostic possibilities based on symptoms."
    )
    self_care: List[str] = Field(
        default_factory=list, description="Conservative home care / OTC guidance."
    )
    see_doctor_if: List[str] = Field(
        default_factory=list, description="Triggers to seek professional care."
    )
    emergency_now_if: List[str] = Field(
        default_factory=list, description="Emergency warning signs requiring immediate care."
    )
    clarifying_questions: List[str] = Field(
        default_factory=list, description="Follow-up questions if key info is missing."
    )
    summary: str = Field(
        ..., description="A short user-facing summary of next steps."
    )


def build_agents_and_manager():
    """Create and initialize AutoGen agents and a GroupChatManager.

    Builds a fresh, isolated set of healthcare triage agents
    (diagnosis, pharmacy, consultation) along with a GroupChatManager.
    A new instance is created per request to prevent cross-session
    state leakage and ensure stable, predictable behavior in a
    multi-user Gradio application.

    Returns:
        A tuple containing:
            patient_agent: Agent used to initiate the consultation.
            manager: GroupChatManager coordinating agent interactions.
    """

    patient_agent = ConversableAgent(
        name="patient",
        system_message="You describe symptoms and ask for medical help.",
        llm_config=llm_config,
        code_execution_config=code_execution_config,
        human_input_mode="NEVER",
    )

    diagnosis_agent = ConversableAgent(
        name="diagnosis",
        system_message=(
            "You analyze symptoms and provide possible causes (not definitive diagnosis). "
            "Ask at most 2 clarifying questions if needed, otherwise summarize key points in ONE response. "
            "Include red-flag symptoms to watch for."
        ),
        llm_config=llm_config,
        code_execution_config=code_execution_config,
        human_input_mode="NEVER",
    )

    pharmacy_agent = ConversableAgent(
        name="pharmacy",
        system_message=(
            "You recommend OTC/self-care options based on the analysis. "
            "Be conservative, include contraindication warnings, and suggest consulting a pharmacist/doctor when relevant. "
            "Only respond once."
        ),
        llm_config=llm_config,
        code_execution_config=code_execution_config,
        human_input_mode="NEVER",
    )

    consultation_agent = ConversableAgent(
    name="consultation",
    system_message=(
        "You are the final triage agent. Produce ONLY a valid JSON object that matches this schema:\n"
        "{\n"
        '  "urgency_level": "none|low|medium|high",\n'
        '  "possible_conditions": ["..."],\n'
        '  "self_care": ["..."],\n'
        '  "see_doctor_if": ["..."],\n'
        '  "emergency_now_if": ["..."],\n'
        '  "clarifying_questions": ["..."],\n'
        '  "summary": "..." \n'
        "}\n\n"
        "Rules:\n"
        "- Output MUST be JSON only (no markdown, no extra text).\n"
        "- Do not diagnose; use cautious language.\n"
        "- Keep items short and actionable.\n"
        "- If any red flags exist, set urgency_level to 'high' and fill emergency_now_if.\n"
        "End."
    ),
    llm_config=llm_config,
    code_execution_config=code_execution_config,
    human_input_mode="NEVER",
)


    groupchat = GroupChat(
        agents=[diagnosis_agent, pharmacy_agent, consultation_agent],
        messages=[],
        max_round=MAX_ROUND,
        speaker_selection_method="round_robin",
    )

    manager = GroupChatManager(name="manager", groupchat=groupchat)

    return patient_agent, manager


def extract_chat_messages(manager):
    """Convert AutoGen group chat messages into Gradio Chatbot format.

    Iterates over messages produced by an AutoGen GroupChatManager and
    transforms them into Gradio-compatible message dictionaries
    (OpenAI-style: {"role", "content"}). Agent names are mapped to
    user-friendly labels for display in the UI.

    Args:
        manager: AutoGen GroupChatManager containing the conversation state.

    Returns:
        A list of dictionaries in Gradio Chatbot "messages" format, where
        each item contains a role ("assistant") and formatted content.
    """

    label_map = {
        "diagnosis": "🩺 Diagnosis Agent",
        "pharmacy": "💊 Pharmacy Agent",
        "consultation": "👨‍⚕️ Consultation Agent",
        "patient": "🧑 Patient",
        "manager": "🧠 Manager",
    }

    messages = []

    for m in getattr(manager.groupchat, "messages", []):
        content = m.get("content", "")
        if not content:
            continue

        name = m.get("name") or m.get("role") or "agent"
        speaker = label_map.get(name, str(name))

        # ✅ REQUIRED FORMAT
        messages.append({
            "role": "assistant",   # Chatbot only supports user/assistant
            "content": f"**{speaker}**\n\n{content}"
        })

    return messages



import traceback

import re
from typing import Dict, Any, List, Tuple
import json
import re
from typing import Tuple

def extract_json(text: str) -> str:
    """Extract the first JSON object from a text string.

    Attempts to locate and return the first valid JSON-like object
    embedded within a larger text response. This is primarily used
    to recover structured outputs from LLM responses when extra
    text is accidentally included.

    Args:
        text: Raw text that may contain an embedded JSON object.

    Returns:
        A string containing the extracted JSON object if found;
        otherwise, an empty string.
    """
    # If agent outputs clean JSON, this returns the whole thing.
    # If not, try to salvage JSON within text.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else ""

def parse_consultation_output(raw_text: str) -> Tuple[Optional[ConsultationOutput], str]:
    """Parse and validate structured consultation output.

    Extracts a JSON object from raw LLM output, parses it, and validates
    it against the ConsultationOutput schema. This ensures the final
    consultation response is well-formed, predictable, and safe for
    downstream use.

    Args:
        raw_text: Raw text output produced by the consultation agent.

    Returns:
        A tuple of:
            parsed: A validated ConsultationOutput instance if parsing
                and validation succeed; otherwise None.
            error_message: An error description if parsing or schema
                validation fails; empty string on success.
    """
    json_str = extract_json(raw_text)
    if not json_str:
        return None, "No JSON object found in consultation output."

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"

    try:
        parsed = ConsultationOutput.model_validate(data)
        return parsed, ""
    except Exception as e:
        return None, f"Schema validation failed: {e}"



def detect_red_flags(symptoms: str, age: str = "", extra: str = "") -> Dict[str, Any]:
    """Detect emergency medical red flags from user-provided text.

    Applies conservative, rule-based pattern matching to identify
    high-risk symptoms (e.g., chest pain, breathing difficulty,
    stroke-like signs). Optionally escalates risk based on age-related
    vulnerability to support early triage decisions.

    This function is designed as a safety pre-check and does not make
    medical diagnoses.

    Args:
        symptoms: Free-text description of current symptoms.
        age: Optional age of the user, used for conservative risk escalation.
        extra: Optional additional medical context or notes.

    Returns:
        A dictionary containing:
            level: Risk level ("none", "medium", or "high").
            flags: List of detected red-flag categories.
            high_risk_age: Whether the user falls into a higher-risk age group.
    """
    text = f"{symptoms or ''} {extra or ''}".lower()

    # Basic normalization
    text = re.sub(r"\s+", " ", text).strip()

    # Patterns: keep conservative + high-signal (avoid over-triggering)
    red_flags: List[Tuple[str, List[str]]] = [
        ("Chest pain / pressure", [
            r"\bchest pain\b", r"\bchest pressure\b", r"\btightness in chest\b",
            r"\bpain in (left )?arm\b", r"\bpain radiat(ing|es)\b"
        ]),
        ("Trouble breathing", [
            r"\bshort(ness)? of breath\b", r"\bdifficulty breathing\b", r"\bcan't breathe\b",
            r"\bwheezing\b", r"\bturn(ing)? blue\b", r"\bbluish\b"
        ]),
        ("Stroke-like symptoms", [
            r"\bface droop\b", r"\bone side\b.*\bweak\b", r"\bsudden weakness\b",
            r"\bsudden numb(ness)?\b", r"\bslurred speech\b", r"\btrouble speaking\b",
            r"\bvision loss\b", r"\bsudden severe headache\b"
        ]),
        ("Fainting / severe confusion", [
            r"\bfaint(ed|ing)?\b", r"\bpassed out\b", r"\bunconscious\b",
            r"\bconfus(ed|ion)\b", r"\bnot responding\b", r"\bseizure\b"
        ]),
        ("Severe bleeding / black stool / vomiting blood", [
            r"\bheavy bleeding\b", r"\bwon't stop bleeding\b",
            r"\bvomit(ing)? blood\b", r"\bblood in vomit\b",
            r"\bblack stool\b", r"\btarry stool\b", r"\bblood in stool\b"
        ]),
        ("Severe allergic reaction", [
            r"\bswelling of (lips|tongue|throat)\b", r"\bthroat swelling\b",
            r"\bhives\b", r"\banaphylaxis\b"
        ]),
        ("High fever with concerning signs", [
            r"\bfever\b.*\bneck stiffness\b", r"\bfever\b.*\brash\b",
        ]),
        ("Suicidal ideation / self-harm", [
            r"\bwant to die\b", r"\bsuicid(al|ideation)\b", r"\bself harm\b"
        ]),
    ]

    hits = []
    for label, patterns in red_flags:
        for p in patterns:
            if re.search(p, text):
                hits.append(label)
                break

    # Age-based conservative escalation
    age_num = None
    try:
        age_num = int(re.sub(r"\D", "", age)) if age else None
    except Exception:
        age_num = None

    high_risk_age = age_num is not None and (age_num < 2 or age_num >= 65)

    level = "none"
    if hits:
        level = "high"
    elif high_risk_age and ("fever" in text or "breathing" in text):
        level = "medium"

    return {
        "level": level,           # "none" | "medium" | "high"
        "flags": sorted(set(hits)),
        "high_risk_age": bool(high_risk_age),
    }


def run_consultation(symptoms: str, age: str, duration: str, extra: str):
    """Run a safety-aware, multi-agent healthcare triage consultation.

    Performs an initial red-flag safety check before orchestrating a
    multi-agent consultation workflow. If high-risk emergency symptoms
    are detected, the agent flow is halted and urgent guidance is
    returned. Otherwise, the function coordinates diagnosis, pharmacy,
    and consultation agents to produce triage-style recommendations.

    This function is intended for educational decision support only and
    does not provide medical diagnoses.

    Args:
        symptoms: Description of the user's symptoms (required).
        age: Optional age of the user for risk context.
        duration: Optional duration of symptoms.
        extra: Optional additional medical context (e.g., conditions,
            medications, allergies).

    Returns:
        A tuple containing:
            chat_messages: List of Gradio-formatted chat messages from agents.
            alert_md: Markdown alert indicating red-flag or caution status.
            final: Final consultation summary and next-step guidance.
            raw_log: Plain-text log of the full agent conversation.
    """
    symptoms = (symptoms or "").strip()
    if not symptoms:
        return [], "", "Please enter symptoms to begin.", ""

    rf = detect_red_flags(symptoms=symptoms, age=age or "", extra=extra or "")

    # 🚨 If high red-flag: stop the agent flow and show urgent guidance
    if rf["level"] == "high":
        flags_text = ", ".join(rf["flags"]) if rf["flags"] else "Emergency warning signs"
        alert_md = (
            "## 🚨 Red-flag detected\n"
            f"**Detected:** {flags_text}\n\n"
            "**This may require urgent medical attention.**\n"
            "- If symptoms are severe or worsening, **call local emergency services** or go to the **nearest emergency department**.\n"
            "- If breathing is difficult, chest pain is severe, or there’s confusion/fainting, **seek immediate help now**.\n\n"
            "_MedAssist MAS provides educational guidance and cannot diagnose emergencies._"
        )
        # Return empty chat + alert + final + raw log
        return [], alert_md, "Red-flag detected — seek urgent medical care.", ""

    # 🟠 Medium: proceed, but show caution banner
    if rf["level"] == "medium":
        alert_md = (
            "### 🟠 Caution\n"
            "Some factors suggest **higher risk** (e.g., age group + symptoms). "
            "Please monitor closely and consider earlier medical advice if symptoms worsen."
        )
    else:
        alert_md = "### ✅ No emergency red-flags detected\nProceeding with triage-style guidance."

    patient_agent, manager = build_agents_and_manager()

    patient_agent.initiate_chat(
        manager,
        message=(
            "I need a medical triage-style suggestion (not a diagnosis). "
            f"Symptoms: {symptoms} | "
            f"Age: {age or 'N/A'} | "
            f"Duration: {duration or 'N/A'} | "
            f"Extra: {extra or 'None'}."
        ),
    )

    chat_messages = extract_chat_messages(manager)

    # Find consultation agent raw message
    consult_raw = ""
    for msg in reversed(chat_messages):
        if "Consultation Agent" in msg["content"]:
            consult_raw = msg["content"]
            break

    parsed, err = parse_consultation_output(consult_raw)

    if parsed:
        final = (
            f"Urgency: {parsed.urgency_level}\n\n"
            f"Summary:\n{parsed.summary}\n\n"
            f"Possible conditions:\n- " + "\n- ".join(parsed.possible_conditions or ["N/A"]) + "\n\n"
            f"Self-care:\n- " + "\n- ".join(parsed.self_care or ["N/A"]) + "\n\n"
            f"See doctor if:\n- " + "\n- ".join(parsed.see_doctor_if or ["N/A"]) + "\n\n"
            f"Emergency now if:\n- " + "\n- ".join(parsed.emergency_now_if or ["N/A"]) + "\n\n"
            f"Clarifying questions:\n- " + "\n- ".join(parsed.clarifying_questions or ["None"])
        )
    else:
        # Fallback to raw text if parsing fails
        final = f"⚠️ Could not parse structured output.\nReason: {err}\n\nRaw output:\n{consult_raw}"

    raw_log = "\n\n".join(m["content"] for m in chat_messages)

    return chat_messages, alert_md, final, raw_log



def clear_all():
    """Reset all Gradio UI outputs and inputs to their default empty state.

    Clears chat history, alert messages, text inputs, and output fields
    to prepare the interface for a new consultation session.

    Returns:
        A tuple of empty values matching the Gradio component order.
    """
    return [], "", "", "", "", "", "", ""


# ---- UI ----
CSS = """
#title { font-size: 28px; font-weight: 800; }
#subtitle { font-size: 14px; opacity: 0.85; }
.gradio-container { max-width: 1050px !important; }
"""

with gr.Blocks(css=CSS, title="MedAssist MAS (Multi-Agent System)") as demo:
    gr.Markdown(
        """
        <div id="title">🧠🩺 AI Healthcare Consultation</div>
        <div id="subtitle">
        Educational triage-style guidance only — not medical advice. If you have severe symptoms (chest pain, trouble breathing, confusion, fainting, severe bleeding),
        seek emergency help immediately.
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            symptoms_in = gr.Textbox(
                label="Describe your symptoms",
                placeholder="e.g., fever 38.5°C, sore throat, cough for 2 days...",
                lines=4,
            )

            with gr.Row():
                age_in = gr.Textbox(label="Age (optional)", placeholder="e.g., 26")
                duration_in = gr.Textbox(label="Duration (optional)", placeholder="e.g., 2 days")

            extra_in = gr.Textbox(
                label="Extra details (optional)",
                placeholder="Allergies, meds, chronic conditions, pregnancy, recent travel, etc.",
                lines=2,
            )

            with gr.Row():
                run_btn = gr.Button("Start Consultation", variant="primary")
                clear_btn = gr.Button("Clear")

            gr.Markdown(
                """
                **Tip:** Provide: temperature, pain scale (0–10), onset, anything that makes it better/worse,
                and any relevant conditions/medications.
                """
            )

        with gr.Column(scale=6):
            alert_out = gr.Markdown()
            chatbot = gr.Chatbot(
                label="Agent Conversation",
                height=420,
                
            )
            final_out = gr.Textbox(
                label="Final Summary & Next Steps",
                lines=8,
            )

            with gr.Accordion("Show raw conversation log", open=False):
                raw_out = gr.Textbox(label="Raw Log", lines=14)

    # Wiring
    run_btn.click(
    fn=run_consultation,
    inputs=[symptoms_in, age_in, duration_in, extra_in],
    outputs=[chatbot, alert_out, final_out, raw_out],
)

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[chatbot, symptoms_in, age_in, duration_in, extra_in, final_out, alert_out, raw_out],
    )

if __name__ == "__main__":
    demo.launch()
