"""
Text utilities for extracting object nouns from action descriptions.

This module provides utilities for parsing VITRA-style action text
to extract object nouns for SAM3D prompts.
"""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# LLM-based object noun extraction (lazy-loaded)
# ---------------------------------------------------------------------------

_llm_extractor = None


class LLMObjectExtractor:
    """Lazy-loaded local LLM for extracting object nouns from action text."""

    _FEW_SHOT_PROMPT = (
        "Extract the main object being held or manipulated by the hand. "
        "Return ONLY the object name, nothing else. "
        "If the action mentions gripping BY a part (e.g. handle, lid, edge), "
        "return the whole object, not the part.\n\n"
        "Action: Pick up the wooden spoon\nObject: wooden spoon\n\n"
        "Action: Put down the container\nObject: container\n\n"
        "Action: Take the pot lid off the pot\nObject: pot lid\n\n"
        "Action: Open the fridge\nObject: fridge\n\n"
        "Action: Wash the plate\nObject: plate\n\n"
        "Action: Peel the potato\nObject: potato\n\n"
        "Action: Stir the soup with a ladle\nObject: ladle\n\n"
        "Action: Pour from the bottle into the cup\nObject: bottle\n\n"
        "Action: Grip and lift the pot by its handle\nObject: pot\n\n"
        "Action: Hold the pan by the edge\nObject: pan\n\n"
        "Action: Carry the bucket by its handle\nObject: bucket\n\n"
    )

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._cache: dict[str, str] = {}

    def _load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"  Loading LLM extractor ({self.model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        print(f"  LLM extractor loaded.")

    def extract(self, text: str) -> Optional[str]:
        """Extract the held/manipulated object noun from *text* using the LLM."""
        if not text or not isinstance(text, str):
            return None

        text = text.strip()
        if text in self._cache:
            return self._cache[text]

        self._load()

        import torch

        prompt = f"{self._FEW_SHOT_PROMPT}Action: {text}\nObject:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Take first line, strip punctuation
        result = generated.split("\n")[0].strip()
        result = re.sub(r"[.,;:!?]+$", "", result).strip()

        if result and len(result) > 1:
            self._cache[text] = result
            return result
        return None

    def check_deformable(self, object_noun: str) -> bool:
        """Return True if the object is deformable (soft, flexible, changes shape)."""
        if not object_noun or not isinstance(object_noun, str):
            return False

        object_noun = object_noun.strip().lower()
        cache_key = f"_deformable_{object_noun}"
        if cache_key in self._cache:
            return self._cache[cache_key] == "yes"

        self._load()

        import torch

        prompt = (
            "Is the following object deformable (soft, flexible, changes shape when handled)? "
            "Answer ONLY 'yes' or 'no'.\n\n"
            "Object: rubber band\nAnswer: yes\n\n"
            "Object: coffee mug\nAnswer: no\n\n"
            "Object: towel\nAnswer: yes\n\n"
            "Object: wooden spoon\nAnswer: no\n\n"
            "Object: sponge\nAnswer: yes\n\n"
            "Object: glass bottle\nAnswer: no\n\n"
            f"Object: {object_noun}\nAnswer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip().lower()

        answer = "yes" if generated.startswith("yes") else "no"
        self._cache[cache_key] = answer
        return answer == "yes"

    def unload(self):
        """Free GPU memory occupied by the LLM."""
        if self.model is not None:
            import torch

            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()
            print("  LLM extractor unloaded.")


def get_llm_extractor(model_name: str = "Qwen/Qwen2.5-7B") -> LLMObjectExtractor:
    """Return the global (lazy-loaded) LLM extractor singleton."""
    global _llm_extractor
    if _llm_extractor is None:
        _llm_extractor = LLMObjectExtractor(model_name)
    return _llm_extractor


def extract_object_noun_llm(text: str) -> Optional[str]:
    """Convenience wrapper: extract object noun using the LLM."""
    return get_llm_extractor().extract(text)


def check_deformable_llm(object_noun: str) -> bool:
    """Convenience wrapper: check if object is deformable using the LLM."""
    return get_llm_extractor().check_deformable(object_noun)


def extract_object_noun(text: str) -> str:
    """
    Extract the object noun from a VITRA action text.

    Parses action descriptions like "Pick up the package of celery" to extract
    just the object noun ("package of celery" or "celery").

    Common patterns:
    - "Pick up the <object>"
    - "Put down the <object>"
    - "Hold the <object>"
    - "Take the <object>"
    - "Move the <object>"
    - "Open the <object>"
    - "Close the <object>"

    Args:
        text: Action description text (e.g., "Pick up the package of celery")

    Returns:
        str: Extracted object noun, or the original text if parsing fails
    """
    if not text or not isinstance(text, str):
        return "object held in hand"

    text = text.strip()

    # Common action verbs that precede objects
    action_patterns = [
        # Pattern: "<verb> the <object>"
        r"(?:pick up|put down|take|hold|grab|move|lift|set down|place|drop|"
        r"open|close|turn on|turn off|switch on|switch off|pour|cut|slice|"
        r"chop|stir|mix|wash|rinse|wipe|dry|peel|grate|squeeze)\s+(?:the\s+)?(.+)",

        # Pattern: "<verb> <object> <preposition> ..."
        r"(?:pick up|put down|take|hold|grab|move|lift|set down|place|drop)\s+"
        r"(.+?)\s+(?:from|into|onto|on|in|to|with|using|near|beside|next to)",

        # Pattern: "...using the <object>"
        r"(?:using|with)\s+(?:the\s+)?(.+?)(?:\s+to|\s*$)",

        # Pattern: simple "the <object>"
        r"^the\s+(.+)$",
    ]

    for pattern in action_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            obj = match.group(1).strip()
            # Remove trailing punctuation
            obj = re.sub(r'[.,;:!?]+$', '', obj)
            # Remove trailing prepositions that got captured
            obj = re.sub(r'\s+(?:from|into|onto|on|in|to|with)$', '', obj, flags=re.IGNORECASE)
            if obj and len(obj) > 1:
                return obj

    # If no pattern matched, try to extract noun phrase after common verbs
    # Split on common verbs and take what follows
    verb_split = re.split(
        r'\b(?:pick up|put down|take|hold|grab|move|lift|set|place|drop|'
        r'open|close|turn|pour|cut|slice|chop|stir|mix|wash|rinse|wipe|'
        r'dry|peel|grate|squeeze)\b',
        text, flags=re.IGNORECASE, maxsplit=1
    )

    if len(verb_split) > 1 and verb_split[1].strip():
        remainder = verb_split[1].strip()
        # Remove leading "the", "a", "an"
        remainder = re.sub(r'^(?:the|a|an)\s+', '', remainder, flags=re.IGNORECASE)
        if remainder:
            return remainder

    # Fallback: return original text, possibly with "the" removed
    result = re.sub(r'^(?:the|a|an)\s+', '', text, flags=re.IGNORECASE)
    return result if result else text


def simplify_object_noun(noun: str) -> str:
    """
    Simplify an object noun by extracting the core object name.

    For example:
    - "package of celery" -> "celery"
    - "box of tissues" -> "tissues"
    - "bottle of water" -> "bottle" (container is important)
    - "red apple" -> "apple"

    Args:
        noun: Object noun string

    Returns:
        str: Simplified object noun
    """
    if not noun:
        return noun

    # Pattern: "<container> of <contents>"
    # For food items, the contents are usually more relevant
    container_pattern = r"(?:package|pack|bag|box|carton|bunch|bundle|pile|stack)\s+of\s+(.+)"
    match = re.search(container_pattern, noun, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # For bottles/cups/glasses, keep the container
    # Pattern: "bottle/cup/glass of <liquid>" -> keep "bottle/cup/glass"
    liquid_container_pattern = r"(bottle|cup|glass|mug|jug|pitcher|can)\s+of\s+.+"
    match = re.search(liquid_container_pattern, noun, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Remove common adjectives that might confuse detection
    adjective_pattern = r"^(?:the|a|an|big|small|large|little|red|blue|green|yellow|white|black|brown|old|new)\s+"
    simplified = re.sub(adjective_pattern, '', noun, flags=re.IGNORECASE)

    return simplified if simplified else noun


def get_detection_prompts(text: str, use_llm: bool = False) -> list:
    """
    Generate multiple detection prompts from an action text.

    Returns a list of prompts from most specific to most general,
    allowing fallback if specific prompts fail.

    Args:
        text: Action description text
        use_llm: If True, use a local LLM for extraction (prepended to cascade)

    Returns:
        list: List of prompt strings, ordered from specific to general
    """
    prompts = []

    if text and text.lower() not in ("object", "object held in hand", "object held in either hand"):
        # LLM-extracted noun is the best prompt (e.g. "pot" from "Grip the pot")
        if use_llm:
            try:
                llm_noun = extract_object_noun_llm(text)
                if llm_noun and llm_noun.lower() != "object held in hand":
                    print(f"  LLM extracted: '{llm_noun}'")
                    prompts.append(llm_noun)
            except Exception as e:
                print(f"  LLM extraction failed ({e}), using regex only")

        # Regex-extracted noun as fallback (e.g. "pot by its handle")
        regex_noun = extract_object_noun(text)
        simplified = simplify_object_noun(regex_noun)
        if simplified and simplified.lower() != regex_noun.lower():
            prompts.append(simplified)
        if regex_noun and regex_noun.lower() != text.lower():
            prompts.append(regex_noun)

        # Full action text last
        prompts.append(text)

    # Generic fallbacks
    prompts.extend([
        "grasped object",
        "held object",
    ])

    # Remove duplicates while preserving order
    seen = set()
    unique_prompts = []
    for p in prompts:
        if p.lower() not in seen:
            seen.add(p.lower())
            unique_prompts.append(p)

    return unique_prompts
