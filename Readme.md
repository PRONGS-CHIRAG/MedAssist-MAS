
---

# 🏥 **MedAssist MAS**

### *A Multi-Agent AI Healthcare Consultation System*

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)
![AutoGen](https://img.shields.io/badge/Agents-AutoGen-green)
![OpenAI](https://img.shields.io/badge/LLM-gpt--4o--mini-purple)
![Status](https://img.shields.io/badge/Status-Demo%20Ready-brightgreen)

---

## 🌟 Overview

**MedAssist MAS (Multi-Agent System)** is an **AI-powered healthcare consultation and triage platform** that simulates a structured medical interaction using **specialized autonomous agents**.

The system provides **educational, triage-style guidance** by:

* Analyzing symptoms
* Suggesting conservative self-care / OTC options
* Advising when professional medical attention is required

> ⚠️ **Disclaimer**
> MedAssist MAS does **NOT** provide medical diagnoses and is **not a replacement for licensed healthcare professionals**.

---

## 🧠 Why MedAssist MAS?

Healthcare systems worldwide face:

* Overloaded primary care
* Unnecessary ER visits
* Lack of early guidance for patients

**MedAssist MAS** demonstrates how **agentic AI architectures** can support:

* Early triage & decision support
* Patient education
* Workflow automation in healthcare settings

All while keeping humans **in the loop**.

---

## 🤖 Multi-Agent Architecture

Each consultation is handled by **role-specific AI agents**, coordinated via an orchestration layer.

| Agent                        | Responsibility                                                       |
| ---------------------------- | -------------------------------------------------------------------- |
| 🧑 **Patient Agent**         | Initiates the consultation using user input                          |
| 🩺 **Diagnosis Agent**       | Analyzes symptoms, suggests possible causes, flags red-warning signs |
| 💊 **Pharmacy Agent**        | Recommends conservative OTC and self-care options                    |
| 👨‍⚕️ **Consultation Agent** | Determines urgency and provides structured next steps with Pydantic output validation                |
| 🧠 **GroupChatManager**      | Controls turn-taking and agent coordination                          |

✔ Fresh agent instances per request
✔ No shared state between users
✔ Safe for concurrent web sessions

---


## 🔐 Safety-First Design: Red-Flag Detection Layer 🚨

MedAssist MAS includes a **pre-agent safety layer** that detects **medical red flags** *before* any AI agents generate guidance.

This ensures that **potential emergencies are escalated immediately**, preventing unsafe or misleading advice.



## 🚨 Red-Flag Detection (New Feature)

Before initiating the multi-agent consultation, MedAssist MAS runs a **rule-based red-flag detection module** that scans user input for high-risk symptoms such as:

* Chest pain or pressure
* Difficulty breathing
* Stroke-like symptoms (slurred speech, weakness, vision loss)
* Fainting or severe confusion
* Severe bleeding or vomiting blood
* Severe allergic reactions (throat/lip swelling)
* High fever with concerning signs
* Suicidal ideation or self-harm language

### How it works

1. User submits symptoms via the Gradio UI
2. **Red-flag detector executes immediately**
3. If **high risk** is detected:

   * Agent flow is **halted**
   * 🚨 Emergency guidance is shown
4. If **medium risk** is detected:

   * A caution banner is displayed
   * Agents proceed conservatively
5. If **no risk** is detected:

   * Full multi-agent consultation runs normally

---

## 🛡️ Safety Levels

| Level         | Behavior                                      |
| ------------- | --------------------------------------------- |
| ✅ **None**    | Proceed with standard triage guidance         |
| 🟠 **Medium** | Show caution banner, advise closer monitoring |
| 🚨 **High**   | Stop agents, instruct urgent medical care     |

This design prevents **hallucinated reassurance** in emergencies and aligns with **responsible AI principles**.



---

## 🏥 Responsible AI & Healthcare Alignment

* Safety checks run **before any LLM reasoning**
* Conservative escalation logic
* No medical diagnoses are made
* Clear emergency disclaimers
* Human-in-the-loop ready

This makes MedAssist MAS suitable for:

* Healthcare AI demos
* Hackathons
* Educational & research prototypes
* Safety-aware agentic systems



---

## 🎨 User Interface

Built with **Gradio**, the UI is designed to be:

* Clean & professional
* Fully non-blocking (no `input()` calls)
* Chat-style agent interaction
* Transparent & debuggable

### UI Components

* 📝 Symptom description input
* 👤 Optional context (age, duration, medical details)
* 💬 Multi-agent conversation view
* ✅ Final consultation summary
* 📜 Raw conversation log (for transparency/debugging)

---

## 🛠️ Tech Stack

* **Python 3.11**
* **Gradio** – Web UI
* **AutoGen** – Multi-agent orchestration
* **OpenAI GPT-4o-mini** – LLM backbone
* **python-dotenv** – Secure configuration

---

## 📂 Project Structure

```
medassist-mas/
│
├── app.py               # Main Gradio application
├── requirements.txt     # Project dependencies
├── .env                 # OPENAI_API_KEY (not committed)
└── README.md            # Documentation
```

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/PRONGS-CHIRAG/MedAssist-MAS.git
cd medassist-mas
```

### 2️⃣ Create & activate a virtual environment

```bash
python -m venv agents_env
source agents_env/bin/activate   # macOS / Linux
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure environment variables

Create a `.env` file:

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 5️⃣ Run the application

```bash
python app.py
```

Open the Gradio URL printed in the terminal 🎉

---

## 🔐 Security & Privacy

* API keys loaded via environment variables
* No persistence of user inputs
* No storage of personal or medical data
* Intended for **demo, research, and educational use**

---

## 🩺 Medical Disclaimer

MedAssist MAS provides **educational triage-style guidance only**.

❌ Not a medical diagnosis
❌ Not a replacement for doctors
❌ Not suitable for emergencies

If you experience **severe or worsening symptoms** (e.g., chest pain, breathing difficulty, confusion, loss of consciousness), seek **immediate medical care**.

---

## 🌍 Use Cases

* 🧪 Hackathons & technical demos
* 🎓 AI/ML portfolios
* 🏥 Healthcare decision-support prototypes
* 🤖 Agentic AI research
* 📊 Human-in-the-loop systems

---

## 🧩 Future Enhancements

* 🎤 Voice-based symptom input (Whisper / ElevenLabs)
* 📄 PDF consultation summary export
* 🧠 RAG with clinical guidelines
* 🔁 Streaming agent responses
* 🛡️ EU-compliant audit logging & disclaimers
* 🧑‍⚕️ Clinician-in-the-loop approval workflows

---

## 👨‍💻 Author

**Chirag Vijay**
AI Engineer | Agentic AI | Healthcare AI | Applied Machine Learning

> Building production-ready AI systems with **multi-agent reasoning**,
> **responsible design**, and **real-world impact**.

---

## ⭐ Support & Contribution

If you find this project useful:

* ⭐ Star the repository
* 🍴 Fork and experiment
* 🧠 Use it as a template for your own agentic systems

---

