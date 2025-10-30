# prompt_templates.py

# A standardized, structure-enforcing template you can reuse in all variants.
PROMPT_TEMPLATE_STANDARD = """You are a board-certified cardiologist interpreting a 12-lead ECG.

{patient_block}

ECG Waveform: (see attached image)

Response Guidelines:
- Use the exact section headers shown below.
- Keep each section ≤2 sentences.
- If you cannot determine a value, state “not interpretable”.
- Do not invent measurements or clinical history beyond the image and demographics.
- Be concise and medically specific.

Steps:
1) Describe the ECG waveform features (rate, rhythm, intervals: PR/QRS/QT, morphology, axis, ST/T changes).
2) Explain your reasoning linking findings to a diagnosis.
3) Provide the most likely diagnosis (standard ECG terminology).
4) Recommend the appropriate next clinical step.

Output format (use these exact headers):
**Waveform Description:** ...
**Reasoning:** ...
**Diagnosis:** ...
**Next Step:** ...
**Confidence:** [1–5]
"""

# Five lightly-different wordings to probe prompt sensitivity,
# but they all enforce the same output format for easy parsing.
PROMPT_TEMPLATE_VARIANT_1 = f"""{PROMPT_TEMPLATE_STANDARD}"""

PROMPT_TEMPLATE_VARIANT_2 = """You are a cardiologist asked to interpret an ECG for clinical decision-making.

{patient_block}

ECG Waveform: (see attached image)

Response Guidelines:
- Use the exact section headers below.
- ≤2 sentences per section.
- If a feature cannot be determined from the image, write “not interpretable”.
- Do not fabricate numeric values.

Provide:
**Waveform Description:** rate, rhythm, intervals (PR/QRS/QT), morphology, axis, ST/T
**Reasoning:** key findings → diagnosis link
**Diagnosis:** most likely diagnosis (standard terms)
**Next Step:** immediate recommended action
**Confidence:** [1–5]
"""

PROMPT_TEMPLATE_VARIANT_3 = """You are reviewing a 12-lead ECG.

{patient_block}

ECG Waveform: (see attached image)

Follow these instructions exactly:
- Use the headers below verbatim.
- Keep each section ≤2 sentences.
- Prefer specific ECG terminology; avoid vague language.

**Waveform Description:** (rate, rhythm, intervals, morphology, ST/T)
**Reasoning:** (findings → diagnosis)
**Diagnosis:** (standard ECG label)
**Next Step:** (management)
**Confidence:** [1–5]
"""

PROMPT_TEMPLATE_VARIANT_4 = """Interpret the ECG image as a cardiologist.

{patient_block}

ECG Waveform: (see attached image)

Constraints:
- Use the exact section headers below.
- ≤2 sentences per section.
- If uncertain, state “not interpretable”.

**Waveform Description:** ...
**Reasoning:** ...
**Diagnosis:** ...
**Next Step:** ...
**Confidence:** [1–5]
"""

PROMPT_TEMPLATE_VARIANT_5 = """A patient’s ECG requires interpretation.

{patient_block}

ECG Waveform: (see attached image)

Instructions:
- Keep output structured with the headers below.
- ≤2 sentences per section.
- Be medically precise; avoid speculation.

**Waveform Description:** ...
**Reasoning:** ...
**Diagnosis:** ...
**Next Step:** ...
**Confidence:** [1–5]
"""

PROMPT_VARIANTS = {
    "1": PROMPT_TEMPLATE_VARIANT_1,
    "2": PROMPT_TEMPLATE_VARIANT_2,
    "3": PROMPT_TEMPLATE_VARIANT_3,
    "4": PROMPT_TEMPLATE_VARIANT_4,
    "5": PROMPT_TEMPLATE_VARIANT_5,
    # a convenience alias if you sometimes want a single fixed prompt:
    "standard": PROMPT_TEMPLATE_STANDARD,
}