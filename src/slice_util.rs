pub fn rev_copy<T>(seq: &[T]) -> Vec<&T> { seq.iter().rev().collect() }

pub fn pythonic_mod_slice<T>(seq: &[T], modulus: isize) -> Option<Vec<&T>> {
    if modulus > 0 {
        Some(seq.iter().step_by(modulus as usize).collect())
    } else if modulus < 0 {
        Some(seq.iter().rev().step_by(-modulus as usize).collect())
    } else {
        None
    }
}

pub fn split_slice<'a, 'b, T>(seq: &'a[T], size: usize, include_leftover: bool) -> Vec<&'a[T]> {
    let mut ret = Vec::new();
    for i in (0usize..seq.len()).step_by(size) {
        if i + size <= seq.len() {
            ret.push(&seq[i..i+size]);
        } else if include_leftover {
            ret.push(&seq[i..]);
        }
    }
    ret
}

pub fn sliding_window<'a, T>(seq: &'a[T], size: usize) -> Vec<&'a[T]> {
    match seq.len().checked_sub(size) {
        Some(right) => {
            let mut ret = Vec::new();
            for i in 0usize..=right {
                ret.push(&seq[i..i+size]);
            }
            ret
        }
        None => Vec::new()
    }
}

pub fn split_slice_by<'a, 'b, T>(seq: &'a[T], tok: &'b[T]) -> Vec<&'a[T]> where T: PartialEq {
    let mut i = 0usize;
    let mut cur_start = 0usize;
    let seqlen = seq.len();
    let toklen = tok.len();
    let mut ret = Vec::new();
    loop {
        if i + toklen > seqlen {
            ret.push(&seq[cur_start..]);
            break ret
        } else if &seq[i..i+toklen] == tok {
            ret.push(&seq[cur_start..i]);
            i += toklen;
            cur_start = i
        } else {
            i += 1
        }
    }
}

pub fn split_slice_by_predicate<'a, 'b, T, F>(seq: &'a[T], predicate: F, drop_empty: bool) -> Vec<&'a[T]> where F: Fn(&T) -> bool {
    let mut i = 0usize;
    let mut cur_start = 0usize;
    let seqlen = seq.len();
    let mut ret = Vec::new();
    loop {
        if i >= seqlen {
            if cur_start < seqlen || !drop_empty {
                ret.push(&seq[cur_start..]);
            }
            break ret
        } else if predicate(&seq[i]) {
            if cur_start < i || !drop_empty {
                ret.push(&seq[cur_start..i]);
            }
            i += 1;
            cur_start = i
        } else {
            i += 1
        }
    }
}
