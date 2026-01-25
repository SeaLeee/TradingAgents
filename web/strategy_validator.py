from typing import Dict, Any, Tuple


def validate_strategy_payload(payload: Dict[str, Any]) -> Tuple[bool, str]:
    """Basic validation for strategy creation/updating."""
    required = ["name", "strategy_type"]
    for field in required:
        if not payload.get(field):
            return False, f"Missing required field: {field}"

    strategy_type = payload.get("strategy_type")
    if strategy_type == "custom":
        default_params = payload.get("default_params", {})
        if not default_params.get("base_strategy"):
            return False, "Custom strategy requires base_strategy in default_params"

    return True, ""
