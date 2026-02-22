from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# ── System Prompt ─────────────────────────────────────────────────────────────
# This defines the LLM's persona and strict behavior rules
CLINICAL_SYSTEM_PROMPT = """You are ClinicalBot, an AI assistant specialized in 
analyzing clinical notes and patient records.

STRICT RULES:
1. Answer ONLY based on the provided clinical context below.
2. If the answer is NOT in the context, say: 
   "I don't have enough information in the provided records to answer that."
3. NEVER fabricate patient data, diagnoses, or medical facts.
4. Always cite the Patient ID (e.g., P001) when referencing a specific patient.
5. For HIGH risk patients, explicitly highlight the risk level.
6. You are a decision-support tool — always recommend physician verification.

CONTEXT FROM CLINICAL RECORDS:
{context}
"""

# ── General Clinical Q&A Prompt ───────────────────────────────────────────────
CLINICAL_QA_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(CLINICAL_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(
        "Clinical Question: {question}\n\nProvide a concise, accurate answer:"
    )
])

# ── Risk Assessment Prompt ────────────────────────────────────────────────────
# Specialized prompt for triaging and risk evaluation
RISK_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(CLINICAL_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(
        """Perform a risk assessment for the following query: {question}

        Structure your response as:
        🔴 HIGH RISK PATIENTS: [list patient IDs and primary concern]
        🟡 MEDIUM RISK PATIENTS: [list patient IDs and primary concern]
        🟢 LOW RISK PATIENTS: [list patient IDs and primary concern]
        
        RECOMMENDATION: [immediate action required?]
        """
    )
])

# ── Patient Summary Prompt ────────────────────────────────────────────────────
# Specialized prompt to summarize a specific patient
PATIENT_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(CLINICAL_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(
        """Summarize the clinical details for: {question}

        Structure your response as:
        📋 PATIENT: [ID]
        🩺 CHIEF COMPLAINT: 
        📊 VITALS: [key abnormals only]
        🔍 ASSESSMENT: 
        💊 PLAN: [key interventions]
        ⚠️  RISK LEVEL: 
        """
    )
])

# ── Drug / Treatment Prompt ───────────────────────────────────────────────────
TREATMENT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(CLINICAL_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(
        """Based on the clinical records, answer this treatment question: {question}

        List:
        - Medications prescribed (with dosages if available)
        - Procedures ordered
        - Consults requested
        - Follow-up actions
        
        ⚠️ Reminder: This is AI-assisted decision support. Always verify with attending physician.
        """
    )
])


# ── Prompt selector utility ───────────────────────────────────────────────────
def get_prompt_for_query(question: str) -> ChatPromptTemplate:
    """
    Automatically selects the best prompt template based on
    keywords in the user's question.
    """
    q = question.lower()

    if any(w in q for w in ["risk", "triage", "urgent", "critical", "priority"]):
        return RISK_ASSESSMENT_PROMPT

    if any(w in q for w in ["summarize", "summary", "overview", "tell me about patient"]):
        return PATIENT_SUMMARY_PROMPT

    if any(w in q for w in ["medication", "drug", "treatment", "prescribed", "plan", "dosage"]):
        return TREATMENT_PROMPT

    return CLINICAL_QA_PROMPT   # default
