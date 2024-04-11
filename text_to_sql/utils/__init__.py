# judge the language of the user input
def is_contain_chinese(user_query: str) -> bool:
    """
    Judge whether the user input contains Chinese characters
    """
    for ch in user_query:
        if "\u4e00" <= ch <= "\u9fff":
            return True

    return False
