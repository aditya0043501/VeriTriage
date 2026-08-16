"""
FastAPI backend for VeriTriage clinical intake tool
Wires together router, extraction, and scoring functions for 3 categories:
  - leg_swelling  (Wells' DVT criteria)
  - sore_throat   (Centor / McIsaac criteria)
  - afib_stroke   (CHA₂DS₂-VASc stroke risk in atrial fibrillation)

Flow:
  1. Routing (first message) — classifies into a category, out_of_scope, or vague.
  2. Vague handling — asks clarifying questions (up to 3 rounds).
  3. Extraction — multi-turn conversation to collect scoring variables.
  4. Deterministic scoring — validated clinical score, never LLM-guessed.
  5. Patient-reported context — optional medications/history as plain text.
  6. Doctor-facing report — structured summary.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("veritriage")

from router import (
    classify_complaint,
    get_out_of_scope_message,
    get_vague_clarifying_question,
    get_vague_escalation_message,
)
from extraction import leg_swelling_extractor, sore_throat_extractor, afib_extractor
from extraction.extraction_utils import detect_category_switch, get_category_switch_message, is_repeat_question
from scoring import calculate_wells_score, calculate_centor_score, calculate_chadsvasc_score
from population_scope import check_population_scope

app = FastAPI(title="VeriTriage API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Category registry ----

def _score_leg_swelling(data):
    return calculate_wells_score(
        active_cancer=data.active_cancer,
        paralysis_or_immobilization=data.paralysis_or_immobilization,
        bedridden_or_surgery=data.bedridden_or_surgery,
        localized_tenderness=data.localized_tenderness,
        entire_leg_swollen=data.entire_leg_swollen,
        calf_swelling_over_3cm=data.calf_swelling_over_3cm,
        pitting_edema=data.pitting_edema,
        collateral_veins=data.collateral_veins,
    )

def _score_sore_throat(data):
    return calculate_centor_score(
        fever=data.fever,
        absence_of_cough=data.absence_of_cough,
        tender_cervical_nodes=data.tender_cervical_nodes,
        tonsillar_exudate=data.tonsillar_exudate,
        age=data.age
    )

def _score_afib_stroke(data):
    return calculate_chadsvasc_score(
        age=data.age,
        sex=data.sex,
        chf_history=data.chf_history,
        hypertension=data.hypertension,
        stroke_tia_history=data.stroke_tia_history,
        vascular_disease=data.vascular_disease,
        diabetes=data.diabetes,
    )

CATEGORY_MODULES = {
    "leg_swelling": {
        "extractor": leg_swelling_extractor,
        "data_class": leg_swelling_extractor.LegSwellingData,
        "score_fn": _score_leg_swelling,
    },
    "sore_throat": {
        "extractor": sore_throat_extractor,
        "data_class": sore_throat_extractor.SoreThroatData,
        "score_fn": _score_sore_throat,
    },
    "afib_stroke": {
        "extractor": afib_extractor,
        "data_class": afib_extractor.AFibStrokeData,
        "score_fn": _score_afib_stroke,
    },
}

MAX_VAGUE_ROUNDS = 3

CATEGORY_NAMES = {
    "leg_swelling": "leg swelling (DVT risk)",
    "sore_throat": "sore throat (strep risk)",
    "afib_stroke": "AFib stroke risk",
}


# ---- Quick-reply chips ----

CHIP_OPTIONS = {
    "yes_no_not_sure": ["Yes", "No", "Not sure"],
    "yes_no": ["Yes", "No"],
    "sex": ["Male", "Female"],
}


def _get_chips(phase: str, current_data: Optional[Dict] = None, category: Optional[str] = None) -> Optional[List[str]]:
    """Return quick-reply chips appropriate to the current conversation phase."""
    if phase == "routing":
        return None
    if phase == "category_switch_pending":
        return CHIP_OPTIONS["yes_no"]
    if phase == "context":
        # Patient-reported medications/history step
        return None
    if phase == "complete":
        return None
    if phase == "extraction" and current_data:
        last_field = current_data.get("last_asked_field")
        if last_field == "sex":
            return CHIP_OPTIONS["sex"]
        if last_field == "age":
            return None
        if last_field in ["afib_confirmed"]:
            return CHIP_OPTIONS["yes_no_not_sure"]
        # All other fields are yes/no-style clinical criteria
        if last_field:
            return CHIP_OPTIONS["yes_no_not_sure"]
    return None


# ---- API models ----

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    current_data: Optional[Dict] = None
    category: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str] = None
    category: Optional[str] = None
    is_complete: bool = False
    current_data: Optional[Dict] = None
    score_result: Optional[Dict] = None
    phase: Optional[str] = None
    patient_context: Optional[str] = None
    doctor_report: Optional[Dict] = None
    chips: Optional[List[str]] = None


conversations: Dict[str, Dict] = {}


@app.get("/")
def read_root():
    return {"message": "VeriTriage API - Clinical Intake Tool", "version": "2.0.0"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


def _build_doctor_report(category, conv_state, score_result, patient_context):
    history = conv_state.get("conversation_history", [])
    chief_complaint = ""
    for turn in history:
        if turn["role"] == "user":
            chief_complaint = turn["content"]
            break

    current_data = conv_state.get("current_data", {})
    scoring_variables = {}
    for key, value in current_data.items():
        if key in ("retry_count", "last_asked_field", "last_bot_message",
                     "history_quality_known", "history_provocation_known",
                     "afib_confirmed"):
            continue
        if value is not None:
            scoring_variables[key] = value

    return {
        "chief_complaint": chief_complaint,
        "category": category.replace("_", " "),
        "scoring_instrument": score_result.get("citation", ""),
        "scoring_variables": scoring_variables,
        "score": score_result.get("score"),
        "tier": score_result.get("tier"),
        "is_partial": score_result.get("isPartial", False),
        "pending_fields": score_result.get("pendingFields", []),
        "breakdown": score_result.get("breakdown", {}),
        "recommendation": score_result.get("recommendation", ""),
        "patient_reported_context": patient_context or "None reported",
        "disclaimer": (
            "This is a structured pre-visit summary generated from patient-reported "
            "symptoms and a validated clinical scoring tool. It is not a diagnosis "
            "and has not been clinically verified."
        ),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    conversation_id = request.conversation_id or f"conv_{len(conversations) + 1}"

    if conversation_id not in conversations:
        conversations[conversation_id] = {
            "category": None,
            "conversation_history": [],
            "current_data": None,
            "is_complete": False,
            "scored": False,
            "vague_rounds": 0,
            "phase": "routing",
            "score_result": None,
            "patient_context": None,
        }
    conv_state = conversations[conversation_id]

    conv_state["conversation_history"].append({"role": "user", "content": request.message})

    def reply(text, category=None, is_complete=False, score_result=None,
              phase=None, patient_context=None, doctor_report=None, chips=None):
        conv_state["conversation_history"].append({"role": "assistant", "content": text})
        response_phase = phase or conv_state.get("phase")
        auto_chips = chips if chips is not None else _get_chips(response_phase, conv_state.get("current_data"), category)
        return ChatResponse(
            response=text,
            conversation_id=conversation_id,
            category=category,
            is_complete=is_complete,
            current_data=conv_state["current_data"],
            score_result=score_result,
            phase=response_phase,
            patient_context=patient_context,
            doctor_report=doctor_report,
            chips=auto_chips,
        )

    # ---- PHASE: routing (with vague clarification) ----
    if conv_state["category"] is None:
        all_user_msgs = [t["content"] for t in conv_state["conversation_history"] if t["role"] == "user"]
        combined_description = " ".join(all_user_msgs)
        category = classify_complaint(combined_description)

        if category == "vague":
            conv_state["vague_rounds"] += 1
            if conv_state["vague_rounds"] >= MAX_VAGUE_ROUNDS:
                conv_state["is_complete"] = True
                conv_state["phase"] = "complete"
                return reply(get_vague_escalation_message(), category="out_of_scope",
                           is_complete=True, phase="complete")
            q = get_vague_clarifying_question(conv_state["vague_rounds"])
            return reply(q, category=None, is_complete=False, phase="routing")

        if category == "out_of_scope":
            conv_state["is_complete"] = True
            conv_state["phase"] = "complete"
            return reply(get_out_of_scope_message(), category="out_of_scope",
                        is_complete=True, phase="complete")

        conv_state["category"] = category
        module = CATEGORY_MODULES[category]
        conv_state["current_data"] = module["data_class"]().model_dump()

        scope_redirect = check_population_scope(request.message)
        if scope_redirect:
            conv_state["is_complete"] = True
            conv_state["phase"] = "complete"
            return reply(scope_redirect, category=category, is_complete=True, phase="complete")

        if conv_state["vague_rounds"] > 0:
            opening = module["extractor"].get_opening_message()
            first_q = module["extractor"].get_initial_question()
            conv_state["phase"] = "extraction"
            return reply(
                f"Thank you for explaining — that helps me understand. {opening} {first_q}",
                category=category, is_complete=False, phase="extraction"
            )

        opening = module["extractor"].get_opening_message()
        first_q = module["extractor"].get_initial_question()
        conv_state["phase"] = "extraction"
        return reply(f"{opening} {first_q}", category=category, is_complete=False, phase="extraction")

    # ---- PHASE: context (patient-reported medications/history) ----
    if conv_state["phase"] == "context":
        context_text = request.message.strip()
        no_markers = ["no", "nope", "nothing", "none", "n/a", "not really",
                       "no nothing", "no nothing else", "nothing else", "i don't have any",
                       "no medications", "no history", "nothing to note", "that's all"]
        if context_text.lower() in no_markers or any(context_text.lower().startswith(m) for m in no_markers):
            conv_state["patient_context"] = "None reported"
        else:
            conv_state["patient_context"] = context_text

        conv_state["phase"] = "complete"
        conv_state["is_complete"] = True

        report = _build_doctor_report(
            conv_state["category"], conv_state,
            conv_state["score_result"], conv_state["patient_context"]
        )

        return reply(
            "Thank you. I've noted that for your doctor to review. Your evaluation is complete — you'll find the full summary and the validated score below.",
            category=conv_state["category"], is_complete=True,
            score_result=conv_state["score_result"],
            phase="complete",
            patient_context=conv_state["patient_context"],
            doctor_report=report
        )

    # ---- PHASE: complete ----
    if conv_state["phase"] == "complete" or conv_state["scored"]:
        return reply(
            "This evaluation is complete. Start a new conversation for another assessment.",
            category=conv_state["category"], is_complete=True, phase="complete"
        )

    # ---- PHASE: category_switch_pending (awaiting confirmation) ----
    if conv_state["phase"] == "category_switch_pending":
        from extraction.rule_fallback import detect_yes_no
        yn = detect_yes_no(request.message)
        requested = conv_state.get("pending_category_switch")
        if yn is True and requested:
            # Perform the switch: reset to the new category
            conv_state["category"] = requested
            conv_state["current_data"] = CATEGORY_MODULES[requested]["data_class"]().model_dump()
            conv_state["phase"] = "extraction"
            conv_state["is_complete"] = False
            conv_state["scored"] = False
            conv_state["score_result"] = None
            conv_state["patient_context"] = None
            conv_state["vague_rounds"] = 0
            new_module = CATEGORY_MODULES[requested]
            opening = new_module["extractor"].get_opening_message()
            first_q = new_module["extractor"].get_initial_question()
            return reply(f"Switched to {CATEGORY_NAMES[requested]}. {opening} {first_q}",
                        category=requested, is_complete=False, phase="extraction")
        elif yn is False:
            conv_state["phase"] = "extraction"
            conv_state["pending_category_switch"] = None
            current = conv_state["category"]
            logger.info(f"[main] Category switch declined; continuing category={current}")
            return reply(f"No problem — we'll continue with your {CATEGORY_NAMES.get(current, current)} assessment. {conv_state.get('last_extraction_question', 'What else can you tell me about your symptoms?')}",
                        category=conv_state["category"], is_complete=False, phase="extraction")
        else:
            return reply(get_category_switch_message(requested, conv_state["category"]),
                        category=conv_state["category"], is_complete=False, phase="category_switch_pending")

    # ---- PHASE: extraction ----
    category = conv_state["category"]
    module = CATEGORY_MODULES[category]
    current_data = module["data_class"](**conv_state["current_data"])

    scope_redirect = check_population_scope(request.message, getattr(current_data, "age", None))
    if scope_redirect:
        conv_state["is_complete"] = True
        conv_state["phase"] = "complete"
        return reply(scope_redirect, category=category, is_complete=True, phase="complete")

    # Category-switch detection: check if the patient is explicitly requesting
    # a different assessment mid-conversation
    if not conv_state.get("scored"):
        switch_target = detect_category_switch(request.message, category)
        if switch_target:
            conv_state["phase"] = "category_switch_pending"
            conv_state["pending_category_switch"] = switch_target
            logger.info(f"[main] Category switch proposed from={category} to={switch_target}")
            return reply(get_category_switch_message(switch_target, category),
                        category=category, is_complete=False, phase="category_switch_pending")

    response, updated_data, is_complete = module["extractor"].extract_and_update_data(
        conversation_history=conv_state["conversation_history"][:-1],
        current_input=request.message,
        current_data=current_data
    )
    conv_state["current_data"] = updated_data.model_dump()
    conv_state["last_extraction_question"] = response

    # TOP-LEVEL CIRCUIT BREAKER: if the extractor's response is a repeat of
    # the last bot message, something went wrong. Force a state check and
    # surface an error rather than looping indefinitely. Skip intentional
    # clarifying re-prompts that embed the original question.
    is_clarifying = response and response.startswith("I'm not sure I understood")
    if response and not is_clarifying and is_repeat_question(response, conv_state["conversation_history"][:-1]):
        # Try to recover by asking about the next missing field deterministically
        missing = updated_data.get_missing_fields() if hasattr(updated_data, "get_missing_fields") else []
        stuck_field = getattr(updated_data, "last_asked_field", None) or "unknown"
        if missing:
            questions_map = {
                "leg_swelling": __import__('extraction.rule_fallback', fromlist=['WELLS_QUESTIONS']).WELLS_QUESTIONS,
                "sore_throat": __import__('extraction.rule_fallback', fromlist=['CENTOR_QUESTIONS']).CENTOR_QUESTIONS,
                "afib_stroke": __import__('extraction.rule_fallback', fromlist=['CHADSVASC_QUESTIONS']).CHADSVASC_QUESTIONS,
            }
            q_map = questions_map.get(category, {})
            next_field = missing[0]
            if next_field in q_map and next_field != stuck_field:
                updated_data.last_asked_field = next_field
                conv_state["current_data"] = updated_data.model_dump()
                cue = " Just one more question." if len(missing) <= 1 else ""
                response = q_map[next_field] + cue
                logger.info(f"[main] TOP-LEVEL CIRCUIT BREAKER recovered: category={category} stuck_field={stuck_field} advanced_to={next_field}")
            else:
                logger.error(f"[main] TOP-LEVEL CIRCUIT BREAKER restart-error: category={category} stuck_field={stuck_field} missing_fields={missing}")
                response = "Something went wrong with this assessment. Let's restart — please describe your main symptom."
                conv_state["phase"] = "complete"
                conv_state["is_complete"] = True
                return reply(response, category=category, is_complete=True, phase="complete")
        else:
            response = "Thank you. I have what I need to complete your assessment."

    if getattr(updated_data, "age", None) is not None and updated_data.age < 18:
        conv_state["is_complete"] = True
        conv_state["phase"] = "complete"
        return reply(check_population_scope("", updated_data.age), category=category,
                    is_complete=True, phase="complete")

    conv_state["is_complete"] = is_complete

    score_result = None
    if is_complete:
        # Guard: if the extractor says complete but the data isn't actually
        # complete (e.g., AFib not confirmed — conversation ends without scoring),
        # end gracefully without attempting to calculate a score.
        if not updated_data.is_complete():
            conv_state["is_complete"] = True
            conv_state["phase"] = "complete"
            return reply(response, category=category, is_complete=True, phase="complete")
        try:
            score_result = module["score_fn"](updated_data)
            if score_result is not None:
                conv_state["scored"] = True
                conv_state["score_result"] = score_result
                conv_state["phase"] = "context"
                conv_state["is_complete"] = False
                return reply(
                    f"{response} Before we finish, are you currently taking any medications, or do you have any other medical history you'd like noted for your doctor? (This won't affect your score — it's just context for your visit.)",
                    category=category, is_complete=False,
                    score_result=score_result, phase="context"
                )
        except Exception as e:
            print(f"Error calculating score: {e}")
            score_result = None
            conv_state["is_complete"] = False
            return reply(
                "I have all your information but ran into a problem calculating your result. Could you try sending your last answer again?",
                category=category, is_complete=False, phase="extraction"
            )

    return reply(response, category=category, is_complete=is_complete,
                score_result=score_result, phase="extraction")


@app.post("/api/reset")
async def reset_conversation(conversation_id: str):
    if conversation_id in conversations:
        del conversations[conversation_id]
    return {"message": "Conversation reset", "conversation_id": conversation_id}


@app.get("/api/conversations")
async def get_conversations():
    return {"conversations": list(conversations.keys())}
