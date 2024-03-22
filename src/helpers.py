def ensure_list(to_list: str | list[str]) -> list[str]:
    """Convert str to list idempotently."""
    return [to_list] if isinstance(to_list, str) else to_list
