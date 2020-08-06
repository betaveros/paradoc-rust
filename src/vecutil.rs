pub fn split_vec_by<'a, 'b, T>(seq: &'a Vec<T>, tok: &'b Vec<T>) -> Vec<&'a [T]> where T: Eq {
    let mut i = 0usize;
    let mut cur_start = 0usize;
    let seqlen = seq.len();
    let toklen = tok.len();
    let mut ret = Vec::new();
    loop {
        if i + toklen > seqlen {
            ret.push(&seq[cur_start..]);
            break ret
        } else if &seq[i..i+toklen] == tok.as_slice() {
            ret.push(&seq[cur_start..i]);
            i += toklen;
            cur_start = i
        } else {
            i += 1
        }
    }
}
