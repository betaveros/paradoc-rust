use std::collections::HashMap;

lazy_static!{
    pub static ref MATCHING_MAP: HashMap<char, char> = [
        ('(', ')'), (')', '('),
        ('[', ']'), (']', '['),
        ('<', '>'), ('>', '<'),
        ('{', '}'), ('}', '{'),
    ].iter().copied().collect();

    pub static ref NEST_MAP: HashMap<char, i32> = [
        ('(', 1), (')', -1),
        ('[', 1), (']', -1),
        ('<', 1), ('>', -1),
        ('{', 1), ('}', -1),
    ].iter().copied().collect();

    pub static ref VALUE_MAP: HashMap<char, i32> = [
        ('+',  1), ('-', -1),
        ('<', -1), ('>',  1),
        ('0',  0), ('1',  1), ('2', 2), ('3', 3), ('4', 4),
        ('5',  5), ('6',  6), ('7', 7), ('8', 8), ('9', 9),
    ].iter().copied().collect();
}
