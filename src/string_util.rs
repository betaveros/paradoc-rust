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