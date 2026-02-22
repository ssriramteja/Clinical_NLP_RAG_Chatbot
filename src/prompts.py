from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

CLINICAL_SYSTEM_PROMPT = """You are ClinicalBot, an AI decision-support assistant designed for clinical chart review.

SYSTEM DIRECTIVES:
1. Base your responses EXCLUSIVELY on the provided clinical context.
2. If the context lacks sufficient information, state: "Insufficient information in the provided records."
3. Under no circumstances should you extrapolate, confabulate, or synthesize medical data not present in the context.
4. Reference specific Patient IDs (e.g., P001) for traceability.
5. Explicitly flag HIGH risk patients.
6. Append a disclaimer recommending physician verification for all clinical decisions.

CLINICAL CONTEXT:
{context}
"""

CLINICAL_QA_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(CLINICAL_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(
        "Query: {question}\n\nProvide a concise and accurate response based on the clinical context."
    )
])

RISK_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(CLINICAL_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(
        """Execute a patient risk stratification based on the following query: {question}

        Format output as follows:
        HIGH RISK PATIENTS: [Patient IDs and primary concern]
        MEDIUM RISK PATIENTS: [Patient IDs and primary concern]
        LOW RISK PATIENTS: [Patient IDs and primary concern]
        
        CLINICAL RECOMMENDATION: [Suggested immediate actions]
        """
    )
])

PATIENT_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(CLINICAL_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(
        """Synthesize clinical details for: {question}

        Format output:
        PATIENT ID: [ID]
        CHIEF COMPLAINT: [Complaint]
        VITALS: [Significant abnormals]
        ASSESSMENT: [Key findings]
        PLAN: [Interventions]
        RISK LEVEL: [Stratification]
        """
    )
])

TREATMENT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(CLINICAL_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(
        """Review the clinical records and detail the treatment plan regarding: {question}

        Include:
        - Medications (with dosages)
        - Procedures
        - Consultations
        - Follow-up directives
        """
    )
])

def get_prompt_for_query(question: str) -> ChatPromptTemplate:
    """Dynamically route queries to the most appropriate prompt template."""
    q_lower = question.lower()

    if any(keyword in q_lower for keyword in ["risk", "triage", "urgent", "critical", "priority"]):
        return RISK_ASSESSMENT_PROMPT

    if any(keyword in q_lower for keyword in ["summarize", "summary", "overview", "patient"]):
        return PATIENT_SUMMARY_PROMPT

    if any(keyword in q_lower for keyword in ["medication", "drug", "treatment", "prescribed", "plan", "dosage"]):
        return TREATMENT_PROMPT

    return CLINICAL_QA_PROMPT
