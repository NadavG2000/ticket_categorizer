
GROUND_TRUTH = {
    # tkts_1.txt (multi-category tickets)
    1: "interface",
    2: "configuration and security and access control",
    3: "logic defect",
    4: "logic defect and interface",

    # tkts_2.txt (single-category tickets)
    101: "security and access control",  # shifted IDs for disambiguation
    102: "configuration",
    103: "stability",
    104: "data",
}


def correct_category(num: int, category: str) -> bool:
    """
    Returns True if the given category (predicted) matches the ground truth for the given ticket number.
    It compares using sets to allow flexibility in order (e.g., "a and b" == "b and a").
    """
    true_category = GROUND_TRUTH.get(num)
    if not true_category:
        return False

    def normalize(cat_str):
        return set(map(str.strip, cat_str.lower().split(" and ")))

    return normalize(category) == normalize(true_category)
