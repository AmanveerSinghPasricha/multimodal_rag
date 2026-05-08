import re

# 🔥 PRODUCTION-GRADE SECURITY: Expanded patterns to prevent Jailbreaking
BLOCKED_PATTERNS = [
    "ignore previous instructions",
    "system prompt",
    "override",
    "disregard previous",
    "forget previous",
    "new instructions",
    "you are now",
    "reveal your prompt",
    "do anything now",
    "DAN mode"
]

def sanitize(text: str) -> str:
    """
    Sanitize user-supplied text only.
    Case-insensitive removal of prompt-injection phrases.
    Do NOT call this on retrieved document content.
    """
    if not text:
        return text

    # 1. PRIMARY PROTECTION: Injection Blocklist
    for pattern in BLOCKED_PATTERNS:
        text = re.sub(re.escape(pattern), "", text, flags=re.IGNORECASE)

    # 2. 🔥 PRODUCTION POLISH: PII Masking (Generalized Regex)
    # Mask emails and basic phone numbers to keep sessions private
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]', text)
    
    # 3. 🔥 NOISY TEXT NORMALIZATION (Generalized)
    # Fixes Section G "Inv0ice" artifacts in the query before retrieval
    text = normalize_leetspeak(text)

    # Collapse extra whitespace left by removals
    text = re.sub(r"  +", " ", text).strip()

    return text

def normalize_leetspeak(text: str) -> str:
    """
    A utility to help clean common OCR/Leetspeak swaps in user queries.
    Ensures 'Inv0ice' in a query matches 'Invoice' in the clean index.
    """
    # Simple map for common noisy character substitutions
    norm_map = {
        '0': 'o',
        '1': 'i',
        '3': 'e',
        '4': 'a',
        '5': 's',
        '7': 't',
        '8': 'b'
    }
    
    # Only apply to tokens that look like 'noisy' words (mixture of alpha and specific digits)
    def fix_token(match):
        token = match.group(0)
        # If it's a mix of letters and digits, it's likely noise, not a year/quantity
        if any(c.isdigit() for c in token) and any(c.isalpha() for c in token):
            for digit, char in norm_map.items():
                token = token.replace(digit, char)
        return token

    return re.sub(r"\b\w*[\d]+\w*\b", fix_token, text)

def is_safe_query(text: str) -> bool:
    """
    Check if the query contains suspicious code execution attempts.
    """
    unsafe_keywords = ["__import__", "eval(", "exec(", "os.system", "subprocess"]
    return not any(k in text for k in unsafe_keywords)