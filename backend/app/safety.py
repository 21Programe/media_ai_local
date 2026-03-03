from __future__ import annotations

import re
from typing import Tuple

# Bloqueios simples (não é perfeito, mas ajuda)
SENSITIVE_PATTERNS = [
    r"\bcpf\b",
    r"\brg\b",
    r"\bcnh\b",
    r"\bpassaporte\b",
    r"\btelefone\b",
    r"\bendereço\b",
    r"\bcep\b",
    r"\bdox\b",
    r"\bdoxxing\b",
    r"\bloc(aliz|alizar|alização)\b",
    r"\brastre(ar|amento)\b",
    r"\bdeepfake\b",
]

def check_prompt(prompt: str) -> Tuple[bool, str]:
    text = (prompt or "").strip().lower()
    if not text:
        return False, "Prompt vazio."

    for pat in SENSITIVE_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            return False, "Pedido bloqueado por envolver dados sensíveis, rastreamento, ou deepfake. Use apenas conteúdo artístico e genérico."

    # bloqueio adicional: "nome completo + documento"
    if "cpf" in text or "rg" in text:
        return False, "Pedido bloqueado por dados sensíveis."

    return True, "ok"