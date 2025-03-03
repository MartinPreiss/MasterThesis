# copy of masterproject

from __future__ import annotations

import re
from collections import defaultdict



def insert_tags(text: str, positions: list[tuple[int, int]], transformation_type:str):
    """
    Inserts multiple tags. Requires that the positions are non-overlapping
    """

    open_tag = f"<{transformation_type}>"
    close_tag = f"</{transformation_type}>"
    offset = 0
    for position in positions:
        start_index = position[0] + offset
        end_index = position[1] + offset
        text = text[0:start_index] + open_tag + text[start_index:end_index] + close_tag + text[end_index:]
        offset += len(open_tag + close_tag)
    return text


def parse_tags(tagged_text: str) -> tuple[str, list[tuple[tuple[int, int], str]]]:
    tag_pattern = r"</?(\w+)[;>]"
    matches = list(re.finditer(tag_pattern, tagged_text))
    tags = []
    tags_with_pos: dict[str, list[int]] = defaultdict(list)

    # Track offset changes due to tag removal
    current_offset = 0
    for match in matches:
        if match.group(0).startswith("</"):  # Closing tag
            type_str = match.group(0)[2:-1]  # Remove '</' and '>'
            if len(tags_with_pos[type_str]) == 0:
                raise ValueError(f"Closing tag '{match.group(0)}' found without a corresponding opening tag")
            start_pos = tags_with_pos[type_str].pop()
            end_pos = match.start() - current_offset

            tags.append(((start_pos, end_pos), type_str))

        else:  # Opening tag
            type_str = match.group(0)[1:-1]  # Remove '<' and '>'
            if type_str not in tags_with_pos:
                tags_with_pos[type_str] = []
            tags_with_pos[type_str].append(match.start() - current_offset)

        tag_length = len(match.group(0))
        current_offset += tag_length

    # Remove all tags from the original text to get clean text
    clean_text = re.sub(tag_pattern, "", tagged_text)
    return clean_text, tags


def ensure_parseable(tagged_text: str) -> bool:
    try:
        parse_tags(tagged_text)
        return True
    except Exception:
        return False


def parse_tag_positions(tagged_text: str) -> tuple[str, list[tuple[int, int]]]:
    try:
        text, positions = parse_tags(tagged_text)
        positions = [p[0] for p in positions]
    except:
        print("parssing positions failed for",tagged_text)
        return (None,None)
    return (text, positions)


def extract_outside_tags(text):
    text_without_tags, tag_positions = parse_tags(text)
    tag_positions = [pos for pos, _ in tag_positions]
    
    result = ""
    last_end = 0
    for start, end in tag_positions:
        result  = result + text_without_tags[last_end:start]
        last_end = end
    
    if last_end != len(text_without_tags):
        result = result + text_without_tags[last_end:]

    return result

def extract_transformation_list(text):
    try: 
        text_without_tags, tag_positions = parse_tags(text)
    except:
        return []
    tag_positions = [pos for pos, _ in tag_positions]
    
    result = []
    for start, end in tag_positions:
        result.append(text_without_tags[start:end])

    return result

def is_outside_tags_equal(original_text_tagged, transformed_text_tagged):
    """Checks if the text outside the tags in the transformed text is equal to the text outside the tags in the original text."""
    original_outside_tags = extract_outside_tags(original_text_tagged)
    transformed_outside_tags = extract_outside_tags(transformed_text_tagged)

    return original_outside_tags == transformed_outside_tags



if __name__ == "__main__":
    text_with_tags = "I like the <swap>sky</swap>. <neg>It is not cool</neg>"
    clean_text, tag_data = parse_tags(text_with_tags)
    for pos, type_ in tag_data:
        print(clean_text, pos, type_)
    
    print(tag_data)
        
    original_text = 'The sky is blue. The see is not green. Apples are always red. Apples are never blue.'
    transformed_text = '<swap>The horizon</swap> is blue. <swap>The sea</swap> is not green. <swap>Pears</swap> are always red. <swap>Pears</swap> are never blue.'
    
    print(is_outside_tags_equal(original_text, transformed_text))