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
            "You decide if a doctor's visit is required and provide a final structured plan:\n"
            "1) What it might be\n"
            "2) What to do now\n"
            "3) When to see a doctor / urgent care / ER\n"
            "4) What info to track\n\n"
            "IMPORTANT: End your response with 'CONSULTATION_COMPLETE'."
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

def run_consultation(symptoms: str, age: str, duration: str, extra: str):
    """Runs a multi-agent AI healthcare triage consultation.

    Validates user input, initializes a fresh AutoGen agent group, and
    orchestrates a structured triage-style conversation (analysis,
    self-care guidance, and next steps). Converts agent messages into
    Gradio-compatible message format and extracts the final summary.

    Disclaimer: This provides educational guidance only and is not a
    medical diagnosis or a substitute for professional care.

    Args:
        symptoms: Description of the user's symptoms (required).
        age: Optional age of the user.
        duration: Optional symptom duration.
        extra: Optional additional medical context.

    Returns:
        A tuple of:
            chat_messages: List of Gradio-formatted chat messages.
            final_summary: Final consultation guidance.
            raw_log: Plain-text conversation log.
    """
    symptoms = (symptoms or "").strip()
    if not symptoms:
        return [], "Please enter symptoms to begin.", ""

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

    # ✅ FIX B: messages format
    chat_messages = extract_chat_messages(manager)

    # Final consultation summary
    final = ""
    for msg in reversed(chat_messages):
        if "Consultation Agent" in msg["content"]:
            final = msg["content"]
            break

    raw_log = "\n\n".join(
        f"{m['content']}" for m in chat_messages
    )

    return chat_messages, final, raw_log



def clear_all():
    return [], "", "", "", "", "", ""


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
        outputs=[chatbot, final_out, raw_out],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[chatbot, symptoms_in, age_in, duration_in, extra_in, final_out, raw_out],
    )

if __name__ == "__main__":
    demo.launch()
