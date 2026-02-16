"""Robust JSON parsing utility for the AI-CoScientist framework.

Handles well-formed JSON, markdown-wrapped JSON, JSON embedded in prose,
and various malformed inputs without ever raising an exception.
"""

import json
import re
from typing import Any, Dict

from loguru import logger


def safely_parse_json(json_str: str) -> Dict[str, Any]:
    """
    Safely parse JSON string, handling potential errors.

    Args:
        json_str: JSON string to parse

    Returns:
        Parsed JSON as dictionary or error dictionary
    """
    if not isinstance(json_str, str):
        logger.error(
            f"Expected string for JSON parsing, got {type(json_str)}"
        )
        return {
            "content": str(json_str),
            "error": f"Invalid input type: {type(json_str)}",
        }

    # Handle empty or whitespace-only strings
    if not json_str.strip():
        logger.warning(
            "Received empty or whitespace-only response from agent"
        )
        return {
            "content": "",
            "error": "Empty response from agent",
        }

    # Strip common markdown code-fence wrappers (``` or ```json)
    cleaned = re.sub(
        r"```(?:json)?\s*([\s\S]*?)```",
        r"\1",
        json_str,
        flags=re.IGNORECASE,
    )
    if cleaned.strip() != json_str.strip():
        logger.debug(
            "Stripped markdown code fences from agent response before JSON parse"
        )
    json_str = cleaned

    # Fast path: attempt full string decode first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass  # Will attempt more robust techniques below
    except Exception as exc:
        logger.error(f"Unexpected error parsing JSON: {exc}")
        return {
            "content": json_str,
            "error": f"Unexpected JSON parse error: {exc}",
        }

    # Technique 1 -- partial decode using JSONDecoder.raw_decode (handles extra data)
    try:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(
            json_str
        )  # Ignore the remainder of the string
        logger.debug(
            "Successfully parsed JSON using raw_decode (partial)"
        )
        return obj if isinstance(obj, dict) else {"content": obj}
    except Exception:
        pass  # Fallthrough to regex extraction

    # Technique 2 -- extract balanced brace substrings
    for i, ch in enumerate(json_str):
        if ch != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for j in range(i, len(json_str)):
            c = json_str[j]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = json_str[i : j + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break  # This '{' failed, try next

    # If all parsing attempts failed, return error with snippet for debugging
    logger.warning(
        f"Failed to parse JSON after multiple attempts. Content snippet: {json_str[:200]}..."
    )
    return {
        "content": json_str,
        "error": "Failed to parse JSON after multiple strategies",
    }
