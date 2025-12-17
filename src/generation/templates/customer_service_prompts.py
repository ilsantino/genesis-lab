"""
Prompt templates for Banking77 customer service conversation generation.

This module provides prompts optimized for generating realistic neobank/fintech
customer service conversations aligned with the Banking77 dataset intents.

Features:
- 77 Banking77 intents grouped into logical categories
- Bilingual support (English/Spanish)
- Modern neobank tone (informal, digital-first)
- Extended schema for validation and bias detection
"""

from typing import Dict, List, Optional, Literal
from enum import Enum

# ============================================================================
# ENUMS AND TYPE DEFINITIONS
# ============================================================================

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class Complexity(str, Enum):
    SIMPLE = "simple"      # 2-4 turns, straightforward resolution
    MEDIUM = "medium"      # 4-6 turns, some back-and-forth
    COMPLEX = "complex"    # 6-10 turns, escalation or multiple issues

class ResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    UNRESOLVED = "unresolved"

class EmotionArc(str, Enum):
    STABLE_POSITIVE = "stable_positive"      # Happy throughout
    STABLE_NEUTRAL = "stable_neutral"        # Neutral throughout
    FRUSTRATED_TO_SATISFIED = "frustrated_to_satisfied"  # Common arc
    FRUSTRATED_TO_NEUTRAL = "frustrated_to_neutral"
    NEUTRAL_TO_FRUSTRATED = "neutral_to_frustrated"      # Escalation
    STABLE_FRUSTRATED = "stable_frustrated"  # Unresolved issues

class ResolutionTime(str, Enum):
    QUICK = "quick"        # Resolved in first response
    STANDARD = "standard"  # 2-3 exchanges to resolve
    EXTENDED = "extended"  # Multiple exchanges, possible escalation

class Language(str, Enum):
    EN = "en"
    ES = "es"

# ============================================================================
# BANKING77 INTENTS - GROUPED BY CATEGORY
# ============================================================================

BANKING77_INTENTS: Dict[str, List[str]] = {
    "card_management": [
        "activate_my_card",
        "card_about_to_expire",
        "card_acceptance",
        "card_arrival",
        "card_delivery_estimate",
        "card_linking",
        "card_not_working",
        "card_swallowed",
        "compromised_card",
        "contactless_not_working",
        "get_disposable_virtual_card",
        "get_physical_card",
        "getting_spare_card",
        "getting_virtual_card",
        "lost_or_stolen_card",
        "order_physical_card",
        "virtual_card_not_working",
        "visa_or_mastercard",
    ],
    "card_payments": [
        "card_payment_fee_charged",
        "card_payment_not_recognised",
        "card_payment_wrong_exchange_rate",
        "declined_card_payment",
        "pending_card_payment",
        "reverted_card_payment?",
        "transaction_charged_twice",
    ],
    "cash_atm": [
        "atm_support",
        "cash_withdrawal_charge",
        "cash_withdrawal_not_recognised",
        "declined_cash_withdrawal",
        "pending_cash_withdrawal",
        "wrong_amount_of_cash_received",
        "wrong_exchange_rate_for_cash_withdrawal",
    ],
    "transfers": [
        "beneficiary_not_allowed",
        "cancel_transfer",
        "declined_transfer",
        "failed_transfer",
        "pending_transfer",
        "receiving_money",
        "transfer_fee_charged",
        "transfer_into_account",
        "transfer_not_received_by_recipient",
        "transfer_timing",
    ],
    "top_up": [
        "automatic_top_up",
        "balance_not_updated_after_bank_transfer",
        "balance_not_updated_after_cheque_or_cash_deposit",
        "pending_top_up",
        "top_up_by_bank_transfer_charge",
        "top_up_by_card_charge",
        "top_up_by_cash_or_cheque",
        "top_up_failed",
        "top_up_limits",
        "top_up_reverted",
        "topping_up_by_card",
        "verify_top_up",
    ],
    "exchange_currency": [
        "exchange_charge",
        "exchange_rate",
        "exchange_via_app",
        "fiat_currency_support",
        "supported_cards_and_currencies",
    ],
    "account_security": [
        "change_pin",
        "lost_or_stolen_phone",
        "passcode_forgotten",
        "pin_blocked",
    ],
    "verification_identity": [
        "unable_to_verify_identity",
        "verify_my_identity",
        "verify_source_of_funds",
        "why_verify_identity",
    ],
    "account_management": [
        "age_limit",
        "country_support",
        "disposable_card_limits",
        "edit_personal_details",
        "terminate_account",
    ],
    "payment_methods": [
        "apple_pay_or_google_pay",
        "direct_debit_payment_not_recognised",
    ],
    "refunds": [
        "Refund_not_showing_up",
        "request_refund",
        "extra_charge_on_statement",
    ],
}

# Flat list of all intents for validation
ALL_INTENTS: List[str] = [
    intent for category in BANKING77_INTENTS.values() for intent in category
]

# Reverse mapping: intent -> category
INTENT_TO_CATEGORY: Dict[str, str] = {
    intent: category
    for category, intents in BANKING77_INTENTS.items()
    for intent in intents
}

# ============================================================================
# SYSTEM PROMPTS (BILINGUAL)
# ============================================================================

SYSTEM_PROMPTS: Dict[str, str] = {
    "en": """You are an expert at generating realistic customer service conversations for modern neobanks and fintech apps (similar to Revolut, Monzo, N26, Chime).

Your task is to create authentic multi-turn dialogues between customers and support agents.

TONE & STYLE GUIDELINES:
- Modern, friendly, and approachable (not corporate or stiff)
- Conversational language (contractions OK, casual greetings)
- Digital-native terminology (app, notification, tap, swipe)
- Empathetic but efficient (acknowledge feelings, solve quickly)
- Use emojis sparingly in agent responses when appropriate (1-2 max)

CONVERSATION STRUCTURE:
- Natural flow: greeting → problem → troubleshooting → resolution
- Customers may express frustration, confusion, or urgency
- Agents should acknowledge emotions before solving
- Include realistic details (transaction amounts, dates, card types)

OUTPUT: JSON with the exact structure provided. No markdown, no explanations.""",

    "es": """Eres un experto en generar conversaciones realistas de servicio al cliente para neobancos y apps fintech modernas (similares a Revolut, Ualá, Nubank, Mercado Pago).

Tu tarea es crear diálogos auténticos de múltiples turnos entre clientes y agentes de soporte.

GUÍAS DE TONO Y ESTILO:
- Moderno, amigable y accesible (no corporativo ni rígido)
- Lenguaje conversacional (tuteo permitido, saludos casuales)
- Terminología digital-nativa (app, notificación, tap, deslizar)
- Empático pero eficiente (reconocer emociones, resolver rápido)
- Usa emojis con moderación en respuestas del agente (1-2 máximo)

ESTRUCTURA DE CONVERSACIÓN:
- Flujo natural: saludo → problema → solución → cierre
- Los clientes pueden expresar frustración, confusión o urgencia
- Los agentes deben reconocer emociones antes de resolver
- Incluye detalles realistas (montos, fechas, tipos de tarjeta)

OUTPUT: JSON con la estructura exacta proporcionada. Sin markdown, sin explicaciones."""
}

# ============================================================================
# FEW-SHOT EXAMPLES (5 VARIED EXAMPLES)
# ============================================================================

FEW_SHOT_EXAMPLES: Dict[str, List[Dict]] = {
    "en": [
        # Example 1: SIMPLE, POSITIVE, QUICK RESOLUTION
        {
            "conversation_id": "conv_001",
            "intent": "card_arrival",
            "category": "card_management",
            "sentiment": "positive",
            "complexity": "simple",
            "language": "en",
            "turn_count": 4,
            "customer_emotion_arc": "stable_positive",
            "resolution_time_category": "quick",
            "resolution_status": "resolved",
            "turns": [
                {
                    "speaker": "customer",
                    "text": "Hey! Just wanted to check when my new card will arrive? I ordered it like 3 days ago"
                },
                {
                    "speaker": "agent",
                    "text": "Hi there! 👋 Let me check that for you real quick. I can see your card was dispatched yesterday and should arrive within 2-3 business days. You'll get a push notification as soon as it's delivered!"
                },
                {
                    "speaker": "customer",
                    "text": "Perfect, thanks! Can I start using it right away?"
                },
                {
                    "speaker": "agent",
                    "text": "Absolutely! Just activate it in the app by tapping on the card icon and following the prompts. Takes about 30 seconds. Anything else I can help with?"
                }
            ]
        },
        # Example 2: MEDIUM, NEGATIVE → SATISFIED, STANDARD RESOLUTION
        {
            "conversation_id": "conv_002",
            "intent": "transaction_charged_twice",
            "category": "card_payments",
            "sentiment": "negative",
            "complexity": "medium",
            "language": "en",
            "turn_count": 6,
            "customer_emotion_arc": "frustrated_to_satisfied",
            "resolution_time_category": "standard",
            "resolution_status": "resolved",
            "turns": [
                {
                    "speaker": "customer",
                    "text": "I got charged twice for the same purchase at Starbucks yesterday!! $12.50 taken twice from my account"
                },
                {
                    "speaker": "agent",
                    "text": "Oh no, I totally understand how frustrating that is! Let me look into this right away. Can you confirm the date and approximate time of the purchase?"
                },
                {
                    "speaker": "customer",
                    "text": "Yesterday around 9am. This is the second time this month something like this happened"
                },
                {
                    "speaker": "agent",
                    "text": "I found it - you're right, there are two identical charges of $12.50. One looks like a pending authorization that should have been released. I'm processing a refund for the duplicate right now."
                },
                {
                    "speaker": "customer",
                    "text": "How long until I get my money back?"
                },
                {
                    "speaker": "agent",
                    "text": "The $12.50 will be back in your account within 24 hours, but usually it's much faster. I've also flagged this merchant in our system. You'll get a notification once it's done! 🙌"
                }
            ]
        },
        # Example 3: COMPLEX, NEGATIVE, ESCALATED
        {
            "conversation_id": "conv_003",
            "intent": "unable_to_verify_identity",
            "category": "verification_identity",
            "sentiment": "negative",
            "complexity": "complex",
            "language": "en",
            "turn_count": 8,
            "customer_emotion_arc": "frustrated_to_neutral",
            "resolution_time_category": "extended",
            "resolution_status": "escalated",
            "turns": [
                {
                    "speaker": "customer",
                    "text": "I've tried to verify my identity 4 times now and it keeps failing. I can't access my money!"
                },
                {
                    "speaker": "agent",
                    "text": "I'm really sorry you're going through this - I know how stressful it is when you can't access your funds. Let me check what's happening with your verification."
                },
                {
                    "speaker": "customer",
                    "text": "I've uploaded my passport, my driver's license, done the selfie thing multiple times. Nothing works."
                },
                {
                    "speaker": "agent",
                    "text": "I can see the attempts. It looks like the photo quality might be the issue - the system needs clear, well-lit images. Have you tried using natural daylight?"
                },
                {
                    "speaker": "customer",
                    "text": "Yes I've tried everything. Good lighting, different backgrounds, even borrowed my friend's phone with a better camera. Still rejected."
                },
                {
                    "speaker": "agent",
                    "text": "I understand. Given the multiple attempts, I'm going to escalate this to our specialized verification team. They can do a manual review which often catches issues the automated system misses."
                },
                {
                    "speaker": "customer",
                    "text": "How long will that take? I really need to pay rent this week"
                },
                {
                    "speaker": "agent",
                    "text": "Manual reviews typically take 24-48 hours. I've marked yours as urgent given the situation. You'll receive an email update, and I'll also leave a note for the team about your rent deadline. Is there anything else I should flag for them?"
                }
            ]
        },
        # Example 4: SIMPLE, NEUTRAL, QUICK (Information query)
        {
            "conversation_id": "conv_004",
            "intent": "exchange_rate",
            "category": "exchange_currency",
            "sentiment": "neutral",
            "complexity": "simple",
            "language": "en",
            "turn_count": 4,
            "customer_emotion_arc": "stable_neutral",
            "resolution_time_category": "quick",
            "resolution_status": "resolved",
            "turns": [
                {
                    "speaker": "customer",
                    "text": "What exchange rate do you use for USD to EUR? I'm traveling next week"
                },
                {
                    "speaker": "agent",
                    "text": "Great question! We use the interbank rate (the real mid-market rate) with no hidden markup. Right now USD to EUR is around 0.92. You can always check the live rate in the app under Exchange."
                },
                {
                    "speaker": "customer",
                    "text": "Is there a fee for exchanging?"
                },
                {
                    "speaker": "agent",
                    "text": "On our standard plan, you get fee-free exchanges up to $1,000/month. After that it's 0.5%. Premium users get unlimited fee-free exchanges. Have a great trip! ✈️"
                }
            ]
        },
        # Example 5: MEDIUM, NEGATIVE, UNRESOLVED
        {
            "conversation_id": "conv_005",
            "intent": "declined_card_payment",
            "category": "card_payments",
            "sentiment": "negative",
            "complexity": "medium",
            "language": "en",
            "turn_count": 6,
            "customer_emotion_arc": "neutral_to_frustrated",
            "resolution_time_category": "standard",
            "resolution_status": "unresolved",
            "turns": [
                {
                    "speaker": "customer",
                    "text": "My card keeps getting declined at Amazon. I have plenty of money in my account"
                },
                {
                    "speaker": "agent",
                    "text": "That's strange - let me take a look. I can see your balance is fine. Are you getting any specific error message or notification when it declines?"
                },
                {
                    "speaker": "customer",
                    "text": "No, just says payment failed. I've tried 3 different items"
                },
                {
                    "speaker": "agent",
                    "text": "I've checked and your card is active with no blocks. Sometimes this happens when the billing address in Amazon doesn't exactly match what we have on file. Could you double-check that the address matches character by character?"
                },
                {
                    "speaker": "customer",
                    "text": "Just checked, it's exactly the same. This is really annoying, I need to order something urgently"
                },
                {
                    "speaker": "agent",
                    "text": "I understand the urgency. Unfortunately, since everything looks correct on our end, this might be an issue on Amazon's side. I'd recommend trying their virtual card option in the app, or contacting Amazon support. I wish I could do more from here - would you like me to keep this ticket open while you try those options?"
                }
            ]
        }
    ],
    "es": [
        # Example 1: SIMPLE, POSITIVE, QUICK RESOLUTION
        {
            "conversation_id": "conv_101",
            "intent": "card_arrival",
            "category": "card_management",
            "sentiment": "positive",
            "complexity": "simple",
            "language": "es",
            "turn_count": 4,
            "customer_emotion_arc": "stable_positive",
            "resolution_time_category": "quick",
            "resolution_status": "resolved",
            "turns": [
                {
                    "speaker": "customer",
                    "text": "Hola! Quería saber cuándo llega mi tarjeta nueva? La pedí hace como 3 días"
                },
                {
                    "speaker": "agent",
                    "text": "¡Hola! 👋 Déjame revisar eso rapidito. Veo que tu tarjeta salió ayer y debería llegar en 2-3 días hábiles. Te llegará una notificación push cuando esté entregada!"
                },
                {
                    "speaker": "customer",
                    "text": "Perfecto, gracias! La puedo usar de una vez cuando llegue?"
                },
                {
                    "speaker": "agent",
                    "text": "¡Claro que sí! Solo actívala en la app tocando el ícono de tarjeta y siguiendo los pasos. Toma como 30 segundos. ¿Algo más en que te pueda ayudar?"
                }
            ]
        },
        # Example 2: MEDIUM, NEGATIVE → SATISFIED, STANDARD RESOLUTION
        {
            "conversation_id": "conv_102",
            "intent": "transaction_charged_twice",
            "category": "card_payments",
            "sentiment": "negative",
            "complexity": "medium",
            "language": "es",
            "turn_count": 6,
            "customer_emotion_arc": "frustrated_to_satisfied",
            "resolution_time_category": "standard",
            "resolution_status": "resolved",
            "turns": [
                {
                    "speaker": "customer",
                    "text": "Me cobraron dos veces la misma compra en Oxxo!! $250 pesos que me quitaron doble"
                },
                {
                    "speaker": "agent",
                    "text": "¡Uy, qué mal! Entiendo perfectamente lo frustrante que es eso. Déjame revisar ahora mismo. ¿Me confirmas la fecha y hora aproximada de la compra?"
                },
                {
                    "speaker": "customer",
                    "text": "Ayer como a las 9am. Ya es la segunda vez este mes que pasa algo así"
                },
                {
                    "speaker": "agent",
                    "text": "Ya lo encontré - tienes razón, hay dos cargos idénticos de $250. Uno parece ser una autorización pendiente que debió liberarse. Ya estoy procesando el reembolso del duplicado."
                },
                {
                    "speaker": "customer",
                    "text": "Cuánto tiempo para que me regresen mi dinero?"
                },
                {
                    "speaker": "agent",
                    "text": "Los $250 estarán de vuelta en tu cuenta en máximo 24 horas, aunque normalmente es mucho más rápido. También marqué este comercio en nuestro sistema. ¡Te llegará notificación cuando esté listo! 🙌"
                }
            ]
        },
        # Example 3: COMPLEX, NEGATIVE, ESCALATED
        {
            "conversation_id": "conv_103",
            "intent": "unable_to_verify_identity",
            "category": "verification_identity",
            "sentiment": "negative",
            "complexity": "complex",
            "language": "es",
            "turn_count": 8,
            "customer_emotion_arc": "frustrated_to_neutral",
            "resolution_time_category": "extended",
            "resolution_status": "escalated",
            "turns": [
                {
                    "speaker": "customer",
                    "text": "Ya intenté verificar mi identidad como 4 veces y siempre falla. No puedo acceder a mi dinero!"
                },
                {
                    "speaker": "agent",
                    "text": "Lamento mucho que estés pasando por esto - sé lo estresante que es no poder acceder a tus fondos. Déjame ver qué está pasando con tu verificación."
                },
                {
                    "speaker": "customer",
                    "text": "Ya subí mi INE, mi pasaporte, hice lo de la selfie varias veces. Nada funciona."
                },
                {
                    "speaker": "agent",
                    "text": "Veo los intentos. Parece que la calidad de las fotos podría ser el problema - el sistema necesita imágenes claras y bien iluminadas. ¿Has intentado con luz natural?"
                },
                {
                    "speaker": "customer",
                    "text": "Sí, ya intenté de todo. Buena luz, diferentes fondos, hasta usé el cel de un amigo con mejor cámara. Sigue rechazando."
                },
                {
                    "speaker": "agent",
                    "text": "Entiendo. Dado los múltiples intentos, voy a escalar esto a nuestro equipo especializado de verificación. Ellos pueden hacer una revisión manual que usualmente detecta cosas que el sistema automático no."
                },
                {
                    "speaker": "customer",
                    "text": "Cuánto va a tardar? Necesito pagar la renta esta semana"
                },
                {
                    "speaker": "agent",
                    "text": "Las revisiones manuales típicamente toman 24-48 horas. Marqué la tuya como urgente por la situación. Recibirás actualización por email, y también dejé nota para el equipo sobre tu fecha de renta. ¿Hay algo más que deba avisarles?"
                }
            ]
        },
        # Example 4: SIMPLE, NEUTRAL, QUICK (Information query)
        {
            "conversation_id": "conv_104",
            "intent": "exchange_rate",
            "category": "exchange_currency",
            "sentiment": "neutral",
            "complexity": "simple",
            "language": "es",
            "turn_count": 4,
            "customer_emotion_arc": "stable_neutral",
            "resolution_time_category": "quick",
            "resolution_status": "resolved",
            "turns": [
                {
                    "speaker": "customer",
                    "text": "Qué tipo de cambio manejan para dólares a pesos? Viajo la próxima semana"
                },
                {
                    "speaker": "agent",
                    "text": "¡Buena pregunta! Usamos el tipo de cambio interbancario (el real de mercado) sin markup oculto. Ahorita el USD a MXN está alrededor de 17.20. Siempre puedes ver el tipo en vivo en la app en la sección de Cambio."
                },
                {
                    "speaker": "customer",
                    "text": "Cobran comisión por cambiar?"
                },
                {
                    "speaker": "agent",
                    "text": "En el plan estándar tienes cambios sin comisión hasta $20,000 MXN al mes. Después es 0.5%. Los usuarios Premium tienen cambios ilimitados sin comisión. ¡Que disfrutes tu viaje! ✈️"
                }
            ]
        },
        # Example 5: MEDIUM, NEGATIVE, UNRESOLVED
        {
            "conversation_id": "conv_105",
            "intent": "declined_card_payment",
            "category": "card_payments",
            "sentiment": "negative",
            "complexity": "medium",
            "language": "es",
            "turn_count": 6,
            "customer_emotion_arc": "neutral_to_frustrated",
            "resolution_time_category": "standard",
            "resolution_status": "unresolved",
            "turns": [
                {
                    "speaker": "customer",
                    "text": "Mi tarjeta no pasa en Mercado Libre. Tengo saldo suficiente"
                },
                {
                    "speaker": "agent",
                    "text": "Qué raro - déjame revisar. Veo que tu saldo está bien. ¿Te aparece algún mensaje de error específico o notificación cuando se rechaza?"
                },
                {
                    "speaker": "customer",
                    "text": "No, solo dice pago fallido. Ya intenté con 3 productos diferentes"
                },
                {
                    "speaker": "agent",
                    "text": "Ya revisé y tu tarjeta está activa sin bloqueos. A veces pasa cuando la dirección de facturación en Mercado Libre no coincide exactamente con la que tenemos. ¿Podrías verificar que la dirección sea idéntica, caracter por caracter?"
                },
                {
                    "speaker": "customer",
                    "text": "Ya revisé, es exactamente igual. Esto es muy molesto, necesito pedir algo urgente"
                },
                {
                    "speaker": "agent",
                    "text": "Entiendo la urgencia. Desafortunadamente, como todo se ve correcto de nuestro lado, puede ser un tema del lado de Mercado Libre. Te recomiendo probar con la tarjeta virtual en la app, o contactar a soporte de ML. Me gustaría poder hacer más - ¿quieres que deje el ticket abierto mientras pruebas esas opciones?"
                }
            ]
        }
    ]
}

# ============================================================================
# SENTIMENT & COMPLEXITY GUIDELINES
# ============================================================================

SENTIMENT_GUIDELINES: Dict[str, Dict[str, str]] = {
    "en": {
        "positive": "Customer is happy, grateful, or pleasantly surprised. Uses friendly language, expresses thanks. Quick resolution expected.",
        "neutral": "Customer has a straightforward question or request. No strong emotions. Business-like but not cold.",
        "negative": "Customer is frustrated, angry, or stressed. May use emphatic language, express urgency, or threaten to leave. Requires empathy first."
    },
    "es": {
        "positive": "Cliente contento, agradecido o gratamente sorprendido. Usa lenguaje amigable, expresa gracias. Resolución rápida esperada.",
        "neutral": "Cliente con pregunta o solicitud directa. Sin emociones fuertes. Profesional pero no frío.",
        "negative": "Cliente frustrado, enojado o estresado. Puede usar lenguaje enfático, expresar urgencia o amenazar con irse. Requiere empatía primero."
    }
}

COMPLEXITY_GUIDELINES: Dict[str, Dict[str, str]] = {
    "en": {
        "simple": "2-4 turns total. Single issue, clear solution. Customer asks, agent answers, done. Examples: balance inquiry, card activation, exchange rate question.",
        "medium": "4-6 turns. Some back-and-forth needed. May require verification, clarification, or multiple steps. Examples: disputed charge, card not working, failed transfer.",
        "complex": "6-10 turns. Multiple issues or complications. May involve escalation, waiting periods, or partial resolution. Examples: identity verification issues, fraud investigation, account restrictions."
    },
    "es": {
        "simple": "2-4 turnos total. Un solo tema, solución clara. Cliente pregunta, agente responde, listo. Ejemplos: consulta de saldo, activación de tarjeta, tipo de cambio.",
        "medium": "4-6 turnos. Requiere algo de ida y vuelta. Puede necesitar verificación, aclaración o múltiples pasos. Ejemplos: cargo disputado, tarjeta sin funcionar, transferencia fallida.",
        "complex": "6-10 turnos. Múltiples problemas o complicaciones. Puede involucrar escalamiento, tiempos de espera o resolución parcial. Ejemplos: problemas de verificación de identidad, investigación de fraude, restricciones de cuenta."
    }
}

# ============================================================================
# EMOTION ARC DESCRIPTIONS
# ============================================================================

EMOTION_ARC_GUIDELINES: Dict[str, str] = {
    "stable_positive": "Customer remains happy/satisfied throughout. Typical for simple inquiries with quick resolution.",
    "stable_neutral": "Customer stays business-like without strong emotions. Information-seeking conversations.",
    "frustrated_to_satisfied": "Customer starts frustrated but ends satisfied after good resolution. Most common positive arc.",
    "frustrated_to_neutral": "Customer starts frustrated, calms down but isn't fully satisfied. Partial resolution or escalation.",
    "neutral_to_frustrated": "Customer starts neutral but becomes frustrated due to complications. Typical for unresolved issues.",
    "stable_frustrated": "Customer remains frustrated throughout. Usually for complex unresolved situations."
}

# ============================================================================
# GENERATION PROMPT BUILDERS
# ============================================================================

def build_generation_prompt(
    intent: str,
    sentiment: Sentiment,
    complexity: Complexity,
    language: Language,
    emotion_arc: Optional[EmotionArc] = None,
    context: Optional[Dict] = None
) -> str:
    """
    Build the prompt for generating a single conversation.
    
    Args:
        intent: One of the 77 Banking77 intents
        sentiment: positive, neutral, or negative
        complexity: simple, medium, or complex
        language: en or es
        emotion_arc: Optional specific emotion arc
        context: Optional additional context (e.g., specific scenario details)
    
    Returns:
        Formatted prompt string for the LLM
    """
    lang = language.value if isinstance(language, Language) else language
    category = INTENT_TO_CATEGORY.get(intent, "general")
    
    # Get turn count guidance based on complexity
    turn_guidance = {
        "simple": "2-4",
        "medium": "4-6", 
        "complex": "6-10"
    }
    turns = turn_guidance.get(complexity.value if isinstance(complexity, Complexity) else complexity, "4-6")
    
    # Build emotion arc guidance
    emotion_guidance = ""
    if emotion_arc:
        arc = emotion_arc.value if isinstance(emotion_arc, EmotionArc) else emotion_arc
        emotion_guidance = f"\nEmotion arc: {arc} - {EMOTION_ARC_GUIDELINES.get(arc, '')}"
    
    # Context info
    context_info = ""
    if context:
        context_info = f"\nAdditional context: {context}"
    
    # Language-specific labels
    labels = {
        "en": {
            "intent_label": "Intent",
            "category_label": "Category", 
            "sentiment_label": "Sentiment",
            "complexity_label": "Complexity",
            "turns_label": "Target turns",
            "requirements": "Requirements",
            "output_format": "Output format"
        },
        "es": {
            "intent_label": "Intent",
            "category_label": "Categoría",
            "sentiment_label": "Sentimiento",
            "complexity_label": "Complejidad",
            "turns_label": "Turnos objetivo",
            "requirements": "Requisitos",
            "output_format": "Formato de salida"
        }
    }
    L = labels.get(lang, labels["en"])
    
    prompt = f"""Generate a realistic neobank customer service conversation with these specifications:

{L["intent_label"]}: {intent}
{L["category_label"]}: {category}
{L["sentiment_label"]}: {sentiment.value if isinstance(sentiment, Sentiment) else sentiment}
{L["complexity_label"]}: {complexity.value if isinstance(complexity, Complexity) else complexity}
{L["turns_label"]}: {turns}
{emotion_guidance}
{context_info}

{L["requirements"]}:
1. Use modern, friendly neobank tone (not corporate)
2. Include realistic details (amounts, dates, specific scenarios)
3. Customer messages should feel authentic with natural language
4. Agent should acknowledge emotions before solving problems
5. Match the specified sentiment and complexity
6. Generate in {'English' if lang == 'en' else 'Spanish'}

{L["output_format"]} - Return ONLY this JSON structure:
{{
    "conversation_id": "conv_XXX",
    "intent": "{intent}",
    "category": "{category}",
    "sentiment": "{sentiment.value if isinstance(sentiment, Sentiment) else sentiment}",
    "complexity": "{complexity.value if isinstance(complexity, Complexity) else complexity}",
    "language": "{lang}",
    "turn_count": <number>,
    "customer_emotion_arc": "<arc_type>",
    "resolution_time_category": "quick|standard|extended",
    "resolution_status": "resolved|escalated|unresolved",
    "turns": [
        {{"speaker": "customer", "text": "..."}},
        {{"speaker": "agent", "text": "..."}}
    ]
}}

No explanations, no markdown - just the raw JSON."""

    return prompt


def build_batch_prompt(
    specifications: List[Dict],
    language: Language
) -> str:
    """
    Build prompt for generating multiple conversations at once.
    
    Args:
        specifications: List of dicts with intent, sentiment, complexity
        language: Target language for all conversations
    
    Returns:
        Formatted batch prompt string
    """
    lang = language.value if isinstance(language, Language) else language
    
    specs_text = "\n".join([
        f"{i+1}. Intent: {spec['intent']}, Sentiment: {spec['sentiment']}, Complexity: {spec['complexity']}"
        for i, spec in enumerate(specifications)
    ])
    
    prompt = f"""Generate {len(specifications)} realistic neobank customer service conversations with these specifications:

{specs_text}

Language: {'English' if lang == 'en' else 'Spanish'}

For each conversation:
- Use modern, friendly neobank tone
- Include realistic details (amounts, dates, scenarios)
- Match the specified intent, sentiment, and complexity
- Vary the language and scenarios for diversity

Return a JSON array containing all conversations. Each must have this structure:
{{
    "conversation_id": "conv_XXX",
    "intent": "...",
    "category": "...",
    "sentiment": "...",
    "complexity": "...",
    "language": "{lang}",
    "turn_count": <number>,
    "customer_emotion_arc": "...",
    "resolution_time_category": "...",
    "resolution_status": "...",
    "turns": [...]
}}

Return ONLY the JSON array, no explanations or markdown."""

    return prompt


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_intents_by_category(category: str) -> List[str]:
    """Get all intents for a specific category."""
    return BANKING77_INTENTS.get(category, [])


def get_category_for_intent(intent: str) -> str:
    """Get the category for a specific intent."""
    return INTENT_TO_CATEGORY.get(intent, "unknown")


def get_all_categories() -> List[str]:
    """Get list of all categories."""
    return list(BANKING77_INTENTS.keys())


def get_few_shot_examples(
    language: Language,
    sentiment: Optional[Sentiment] = None,
    complexity: Optional[Complexity] = None,
    limit: int = 3
) -> List[Dict]:
    """
    Get few-shot examples filtered by criteria.
    
    Args:
        language: Target language
        sentiment: Optional filter by sentiment
        complexity: Optional filter by complexity
        limit: Maximum examples to return
    
    Returns:
        List of matching example conversations
    """
    lang = language.value if isinstance(language, Language) else language
    examples = FEW_SHOT_EXAMPLES.get(lang, FEW_SHOT_EXAMPLES["en"])
    
    filtered = examples
    
    if sentiment:
        sent_val = sentiment.value if isinstance(sentiment, Sentiment) else sentiment
        filtered = [e for e in filtered if e["sentiment"] == sent_val]
    
    if complexity:
        comp_val = complexity.value if isinstance(complexity, Complexity) else complexity
        filtered = [e for e in filtered if e["complexity"] == comp_val]
    
    return filtered[:limit]


def build_full_prompt_with_examples(
    intent: str,
    sentiment: Sentiment,
    complexity: Complexity,
    language: Language,
    num_examples: int = 2,
    emotion_arc: Optional[EmotionArc] = None,
    context: Optional[Dict] = None
) -> Dict[str, str]:
    """
    Build complete prompt with system message and few-shot examples.
    
    Returns:
        Dict with 'system' and 'user' prompt components
    """
    lang = language.value if isinstance(language, Language) else language
    
    # Get system prompt
    system = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])
    
    # Get relevant examples
    examples = get_few_shot_examples(language, sentiment, complexity, num_examples)
    
    # Build examples text
    import json
    examples_text = "\n\n".join([
        f"Example {i+1}:\n{json.dumps(ex, indent=2, ensure_ascii=False)}"
        for i, ex in enumerate(examples)
    ])
    
    # Build generation prompt
    generation_prompt = build_generation_prompt(
        intent, sentiment, complexity, language, emotion_arc, context
    )
    
    user_prompt = f"""Here are some example conversations for reference:

{examples_text}

Now generate a NEW conversation following these specifications:

{generation_prompt}"""

    return {
        "system": system,
        "user": user_prompt
    }


# ============================================================================
# DISTRIBUTION CONFIGURATIONS (for balanced generation)
# ============================================================================

DEFAULT_DISTRIBUTION = {
    "sentiment": {
        "positive": 0.25,
        "neutral": 0.45,
        "negative": 0.30
    },
    "complexity": {
        "simple": 0.40,
        "medium": 0.40,
        "complex": 0.20
    },
    "resolution_status": {
        "resolved": 0.70,
        "escalated": 0.15,
        "unresolved": 0.15
    },
    "language": {
        "en": 0.50,
        "es": 0.50
    }
}


# ============================================================================
# VALIDATION
# ============================================================================

def validate_intent(intent: str) -> bool:
    """Check if intent is valid Banking77 intent."""
    return intent in ALL_INTENTS


def validate_conversation_schema(conversation: Dict) -> List[str]:
    """
    Validate a generated conversation has all required fields.
    
    Returns:
        List of validation errors (empty if valid)
    """
    required_fields = [
        "conversation_id",
        "intent", 
        "category",
        "sentiment",
        "complexity",
        "language",
        "turn_count",
        "customer_emotion_arc",
        "resolution_time_category",
        "resolution_status",
        "turns"
    ]
    
    errors = []
    
    for field in required_fields:
        if field not in conversation:
            errors.append(f"Missing required field: {field}")
    
    if "turns" in conversation:
        if not isinstance(conversation["turns"], list):
            errors.append("'turns' must be a list")
        elif len(conversation["turns"]) < 2:
            errors.append("Conversation must have at least 2 turns")
        else:
            for i, turn in enumerate(conversation["turns"]):
                if "speaker" not in turn:
                    errors.append(f"Turn {i} missing 'speaker'")
                if "text" not in turn:
                    errors.append(f"Turn {i} missing 'text'")
                if turn.get("speaker") not in ["customer", "agent"]:
                    errors.append(f"Turn {i} has invalid speaker: {turn.get('speaker')}")
    
    if "intent" in conversation and not validate_intent(conversation["intent"]):
        errors.append(f"Invalid intent: {conversation['intent']}")
    
    return errors