import json
import os
import time
from typing import Any, Dict, Optional

from common_utils import get_response


LOG_FILE = "scratchpad_logs.json"


def _current_ts() -> str:
    return str(int(time.time() * 1000))


def wrap_with_scratchpad_instruction(query: str) -> list[Dict[str, str]]:
    """Return a messages list that instructs the LLM to produce a scratchpad plus final answer.
    The LLM is asked to output clearly delimited sections.
    We attempt to parse these markers. If they are missing, the parser will return the raw
    text under `final` and keep the entire response as `raw`.
    """
    # If original_messages is a list of messages, preserve them after a system instruction.
    system_instruction = (
        "You are going to produce a short scratchpad and then a final concise answer.\n"
        "Format your output EXACTLY as follows (use the markers):\n\n"
        "<think>\n"
        "WHAT_I_KNOW:\n<bullet points or short sentences>\n\n"
        "WHAT_I_NEED:\n<bullet points or short sentences>\n\n"
        "WHAT_I_AM_ASSUMING:\n<bullet points or short sentences>\n"
        "</think>\n"
        "---FINAL_ANSWER_START---\n"
        "<One concise final paragraph that answers the user. This is the only text shown to the user.>\n"
        "---FINAL_ANSWER_END---\n\n"
        "Do not include any other text outside these markers. Keep the scratchpad truthful about uncertainty."
    )

    messages_out: list[Dict[str, str]] = [{"role": "system", "content": system_instruction}]

    if isinstance(query, str):
        messages_out.append({"role": "user", "content": query})
    else:
        # assume it's a list of messages; append them in order
        for m in query:
            # sanitize to basic dict with role/content
            if isinstance(m, dict) and "role" in m and "content" in m:
                messages_out.append({"role": m["role"], "content": m["content"]})
            else:
                messages_out.append({"role": "user", "content": str(m)})

    return messages_out


def parse_scratchpad_response(response_text: str) -> Dict[str, Optional[str]]:
    """Parse the response text into scratchpad parts and final answer.

    Returns dict with keys: what_i_know, what_i_need, what_i_am_assuming, final, raw
    """
    out = {
        "what_i_know": None,
        "what_i_need": None,
        "what_i_am_assuming": None,
        "final": None,
        "raw": response_text,
    }

    # Allow models to place the scratchpad inside <think>...</think> tags.
    text_to_search = response_text
    if "<think>" in response_text and "</think>" in response_text:
        try:
            tstart = response_text.index("<think>") + len("<think>")
            tend = response_text.index("</think>", tstart)
            text_to_search = response_text[tstart:tend].strip()
        except ValueError:
            text_to_search = response_text

    try:
        # Prefer explicit SCRATCHPAD markers if present
        if "---SCRATCHPAD_START---" in text_to_search and "---SCRATCHPAD_END---" in text_to_search:
            start_sp = text_to_search.index("---SCRATCHPAD_START---")
            end_sp = text_to_search.index("---SCRATCHPAD_END---", start_sp)
            sp_block = text_to_search[start_sp + len("---SCRATCHPAD_START---") : end_sp].strip()
        else:
            # fallback: assume entire think block is the scratchpad
            sp_block = text_to_search

        def extract_section(label: str, text: str) -> Optional[str]:
            key = label + ":"
            if key in text:
                try:
                    start = text.index(key) + len(key)
                    # determine next possible labels (order matters)
                    next_labels = ["WHAT_I_NEED:", "WHAT_I_AM_ASSUMING:"]
                    # find the earliest occurrence of any next label after start
                    next_positions = [text.index(l, start) for l in next_labels if l in text[start:]]
                    next_pos = min(next_positions) if next_positions else len(text)
                    return text[start:next_pos].strip()
                except Exception:
                    return None
            return None

        out["what_i_know"] = extract_section("WHAT_I_KNOW", sp_block)
        out["what_i_need"] = extract_section("WHAT_I_NEED", sp_block)
        out["what_i_am_assuming"] = extract_section("WHAT_I_AM_ASSUMING", sp_block)

        # final answer can be outside or after the think tags; search full response_text
        if "---FINAL_ANSWER_START---" in response_text and "---FINAL_ANSWER_END---" in response_text:
            fstart = response_text.index("---FINAL_ANSWER_START---") + len("---FINAL_ANSWER_START---")
            fend = response_text.index("---FINAL_ANSWER_END---", fstart)
            out["final"] = response_text[fstart:fend].strip()
        else:
            # if no explicit final markers, try to find an ending paragraph after the think block
            # naive heuristic: the text after </think> if present
            if "</think>" in response_text:
                try:
                    after = response_text.split("</think>", 1)[1].strip()
                    # take the first non-empty paragraph as final
                    paras = [p.strip() for p in after.splitlines() if p.strip()]
                    if paras:
                        out["final"] = paras[0]
                except Exception:
                    pass
    except Exception:
        pass

    return out


def log_scratchpad(cache_key: str, parsed: Dict[str, Optional[str]],
                   extra: Optional[Dict[str, Any]] = None) -> None:
    entry = {
        "ts": _current_ts(),
        "cache_key": cache_key,
        "parsed": parsed,
    }
    if extra:
        entry["extra"] = extra

    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []

    logs.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def get_scratchpad_response(query: str, **kwargs) -> str:
    wrapped = wrap_with_scratchpad_instruction(query)
    response = get_response(wrapped, **kwargs)
    if not response:
        return ""

    parsed = parse_scratchpad_response(response)
    final = parsed.get("final") or parsed.get("raw") or ""
    return final

if __name__ == "__main__":
	# Example usage
	test_query = "Is it possible for water to flow uphill?"
	response = get_scratchpad_response(test_query)
	print("Final Answer:\n", response)
