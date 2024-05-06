"""Some utility functions for text to SQL."""

# judge the language of the user input
from typing import Tuple


def is_contain_chinese(user_query: str) -> bool:
    """
    Judge whether the user input contains Chinese characters
    """
    for ch in user_query:
        if "\u4e00" <= ch <= "\u9fff":
            return True

    return False


def find_bracket_index(input_str: str, start_index_to_find: int = 0) -> Tuple[int, int]:
    """
    Find the start and end index of the bracket in the input string

    Args:
        input_str (str): input string
        start_index_to_find (int): start index to find the bracket, default is 0

    Returns:
        Tuple[int, int], the first element is the start index of the bracket,
        the second element is the end index of the bracket
    """
    start_index = input_str.find("{", start_index_to_find)
    end_index = input_str.find("}", start_index + 1)
    if start_index == -1 or end_index == -1:
        return -1, -1

    return start_index, end_index
