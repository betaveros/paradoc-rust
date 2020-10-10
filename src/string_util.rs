use regex::Regex;
use num::bigint::BigInt;

// Expand a string of character ranges.
// Example: str_class("a-cx0-9") = "abcx0123456789"
pub fn str_class(s: &str) -> String {
    let mut ret = String::new();
    let mut it = s.chars();

    let mut c1 = it.next();
    let mut c2 = it.next();
    let mut c3 = it.next();

    loop {
        match (c1, c2, c3) {
            (Some(start), Some('-'), Some(end)) => {
                if start <= end {
                    ret.extend(start..=end);
                } else {
                    ret.extend((end..=start).rev());
                }
                c1 = it.next();
                c2 = it.next();
                c3 = it.next();
            }
            (Some(c), _, _) => {
                ret.push(c);
                c1 = c2;
                c2 = c3;
                c3 = it.next();
            }
            (None, _, _) => break,
        }
    }

    ret.shrink_to_fit();
    ret
}

pub fn int_groups(text: &str) -> impl Iterator<Item=BigInt> + '_ {
    lazy_static! {
        static ref INT_PATTERN: Regex = Regex::new(r#"-?\d+"#).unwrap();
    }
    INT_PATTERN.find_iter(text).map(|m| m.as_str().parse::<BigInt>().unwrap())
}

pub fn float_groups(text: &str) -> impl Iterator<Item=f64> + '_ {
    lazy_static! {
        static ref INT_PATTERN: Regex = Regex::new(r#"-?\d+(?:\.\d+)?(?:e\d+)?|\.\d+(?:e\d+)?"#).unwrap();
    }
    INT_PATTERN.find_iter(text).map(|m| m.as_str().parse::<f64>().unwrap())
}


// Naively flatmapping with .chars() doesn't work because the Chars is only as alive as the String
// it's based on (rustc --explain E0515)
// https://stackoverflow.com/questions/64228432/flatten-iterator-of-strings-to-vecchar-or-even-fromiteratorchar
pub fn flat_collect_strs<S>(ss: impl Iterator<Item=String>) -> S where S: Default + Extend<char> {
    // https://stackoverflow.com/questions/47193584/is-there-an-owned-version-of-stringchars
    // returns S: FromIterator<char> with extra allocation
    // ss.flat_map(|s| s.chars().collect::<Vec<char>>().into_iter()).collect()
    let mut acc: S = Default::default();
    for s in ss {
        acc.extend(s.chars());
    }
    acc
}
