use std::cmp::Ordering;
use std::hash::Hash;
use std::collections::{HashSet, HashMap};
use std::mem;


// Generic over I=PdObj and K=PdKey.

fn key_counter<I, K: Hash + Eq, E, F>(a: impl IntoIterator<Item=I>, mut proj: F) -> Result<HashMap<K, usize>, E> where F: FnMut(&I) -> Result<K, E> {
    let mut ret = HashMap::new();
    for e in a {
        let key = proj(&e)?;
        match ret.get_mut(&key) {
            Some(place) => { *place = *place + 1usize; }
            None => { ret.insert(key, 1usize); }
        }
    }
    Ok(ret)
}

fn key_counter_and_collect<I, K: Hash + Eq, E, F>(a: impl IntoIterator<Item=I>, mut proj: F) -> Result<(HashMap<K, usize>, Vec<I>), E> where F: FnMut(&I) -> Result<K, E> {
    let mut ret = HashMap::new();
    let mut acc = Vec::new();
    for e in a {
        let key = proj(&e)?;
        match ret.get_mut(&key) {
            Some(place) => { *place = *place + 1usize; }
            None => { ret.insert(key, 1usize); }
        }

        acc.push(e);
    }
    Ok((ret, acc))
}

fn key_set<I, K: Hash + Eq, E, F>(a: impl IntoIterator<Item=I>, mut proj: F) -> Result<HashSet<K>, E> where F: FnMut(&I) -> Result<K, E> {
    let mut ret = HashSet::new();
    for e in a {
        ret.insert(proj(&e)?);
    }
    Ok(ret)
}

pub fn intersection<I, K: Hash + Eq, E, F>(a: impl IntoIterator<Item=I>, b: impl IntoIterator<Item=I>, mut proj: F) -> Result<Vec<I>, E> where F: FnMut(&I) -> Result<K, E> {
    let mut counter = key_counter(b, &mut proj)?;
    let mut acc = Vec::new();
    for e in a {
        let key = proj(&e)?;
        match counter.get_mut(&key) {
            Some(place) => { if *place > 0 { acc.push(e); *place -= 1; } }
            None => {}
        }
    }
    Ok(acc)
}

pub fn union<I, K: Hash + Eq, E, F>(a: impl IntoIterator<Item=I>, b: impl IntoIterator<Item=I>, mut proj: F) -> Result<Vec<I>, E> where F: FnMut(&I) -> Result<K, E> {
    let (mut counter, mut acc) = key_counter_and_collect(a, &mut proj)?;
    for e in b {
        let key = proj(&e)?;
        match counter.get_mut(&key) {
            Some(place) => {
                if *place > 0 { *place -= 1; } else { acc.push(e); }
            }
            None => {
                acc.push(e);
            }
        }
    }
    Ok(acc)
}

pub fn set_difference<I, K: Hash + Eq, E, F>(a: impl IntoIterator<Item=I>, b: impl IntoIterator<Item=I>, mut proj: F) -> Result<Vec<I>, E> where F: FnMut(&I) -> Result<K, E> {
    let ks = key_set(b, &mut proj)?;
    let mut acc = Vec::new();
    for e in a {
        if !ks.contains(&proj(&e)?) {
            acc.push(e);
        }
    }
    Ok(acc)
}

pub fn symmetric_difference<I, K: Hash + Eq, E, F>(a: impl IntoIterator<Item=I> + Clone, b: impl IntoIterator<Item=I> + Clone, mut proj: F) -> Result<Vec<I>, E> where F: FnMut(&I) -> Result<K, E> {
    let mut acc = Vec::new();

    let bks = key_set(b.clone(), &mut proj)?;
    for e in a.clone() {
        if !bks.contains(&proj(&e)?) { acc.push(e); }
    }

    let aks = key_set(a, &mut proj)?;
    for e in b {
        if !aks.contains(&proj(&e)?) { acc.push(e); }
    }

    Ok(acc)
}

// ---

fn force_keyed_vec<I, K, E, F>(it: impl IntoIterator<Item=I>, mut proj: F) -> Result<Vec<(K, I)>, E> where F: FnMut(&I) -> Result<K, E> {
    // things like sort_by_cached_key are not enough because we also want to force the PdResult
    it.into_iter().map(|e| Ok((proj(&e)?, e))).collect::<Result<Vec<(K, I)>, E>>()
}

pub fn organize_by<I, K: Clone + Hash + Eq, E, F>(it: impl IntoIterator<Item=I>, mut proj: F) -> Result<Vec<Vec<I>>, E> where F: FnMut(&I) -> Result<K, E> {
    let mut groups: Vec<Vec<I>> = Vec::new();
    let mut group_indices: HashMap<K, usize> = HashMap::new();

    for e in it {
        let key = proj(&e)?;
        match group_indices.get(&key) {
            Some(i) => { groups[*i].push(e); }
            None => {
                group_indices.insert(key, groups.len());
                groups.push(vec![e]);
            }
        }
    }
    Ok(groups)
}

pub fn sort_by<I, K: Ord, E, F>(it: impl IntoIterator<Item=I>, proj: F) -> Result<Vec<I>, E> where F: FnMut(&I) -> Result<K, E> {
    let mut keyed_vec = force_keyed_vec(it, proj)?;
    // sort_by_key's key function wants us to give ownership of the key :-/
    keyed_vec.sort_by(|x, y| x.0.cmp(&y.0));

    Ok(keyed_vec.into_iter().map(|x| x.1).collect())
}

pub fn max_by<I, K: Ord, E, F>(it: impl IntoIterator<Item=I>, proj: F) -> Result<Option<I>, E> where F: FnMut(&I) -> Result<K, E> {
    let keyed_vec = force_keyed_vec(it, proj)?;
    Ok(keyed_vec.into_iter().max_by(|x, y| x.0.cmp(&y.0)).map(|x| x.1))
}

pub fn min_by<I, K: Ord, E, F>(it: impl IntoIterator<Item=I>, proj: F) -> Result<Option<I>, E> where F: FnMut(&I) -> Result<K, E> {
    let keyed_vec = force_keyed_vec(it, proj)?;
    Ok(keyed_vec.into_iter().min_by(|x, y| x.0.cmp(&y.0)).map(|x| x.1))
}

pub fn maxima_by<I, K: Ord, E, F>(it: impl IntoIterator<Item=I>, proj: F) -> Result<Vec<I>, E> where F: FnMut(&I) -> Result<K, E> {
    let keyed_vec = force_keyed_vec(it, proj)?;
    let mut best: Option<(K, Vec<I>)> = None;
    for (k, v) in keyed_vec {
        match &mut best {
            None => { best = Some((k, vec![v])); }
            Some((bk, bvs)) => match k.cmp(bk) {
                Ordering::Greater => { best = Some((k, vec![v])); }
                Ordering::Equal => { bvs.push(v); }
                Ordering::Less => {}
            }
        }
    }
    Ok(best.map_or_else(Vec::new, |(_, bvs)| bvs))
}

pub fn minima_by<I, K: Ord, E, F>(it: impl IntoIterator<Item=I>, proj: F) -> Result<Vec<I>, E> where F: FnMut(&I) -> Result<K, E> {
    let keyed_vec = force_keyed_vec(it, proj)?;
    let mut best: Option<(K, Vec<I>)> = None;
    for (k, v) in keyed_vec {
        match &mut best {
            None => { best = Some((k, vec![v])); }
            Some((bk, bvs)) => match k.cmp(bk) {
                Ordering::Less => { best = Some((k, vec![v])); }
                Ordering::Equal => { bvs.push(v); }
                Ordering::Greater => {}
            }
        }
    }
    Ok(best.map_or_else(Vec::new, |(_, bvs)| bvs))
}

// TODO if this stabilizes https://github.com/rust-lang/rust/issues/53485
pub fn is_sorted_by<I, K: Ord, E, F>(it: impl IntoIterator<Item=I>, mut proj: F, accept: fn(Ordering) -> bool) -> Result<bool, E> where F: FnMut(&I) -> Result<K, E> {
    let mut prev: Option<K> = None;
    for e in it {
        let cur = proj(&e)?;
        if let Some(p) = prev {
            if !accept(p.cmp(&cur)) {
                return Ok(false)
            }
        }
        prev = Some(cur);
    }
    Ok(true)
}

// TODO: we don't need to hash here, this could take a PdObj projection, but I'm too lazy
pub fn group_by<I, K: Hash + Eq, E, F>(it: impl IntoIterator<Item=I>, mut proj: F) -> Result<Vec<Vec<I>>, E> where F: FnMut(&I) -> Result<K, E> {
    let mut acc: Vec<Vec<I>> = Vec::new();
    let mut cur: Option<(K, Vec<I>)> = None;

    for e in it {
        let key = proj(&e)?;
        match &mut cur {
            Some((cur_key, cur_group)) => {
                if cur_key == &key {
                    cur_group.push(e);
                } else {
                    *cur_key = key;
                    acc.push(mem::replace(cur_group, vec![e]));
                }
            }
            None => {
                cur = Some((key, vec![e]));
            }
        }
    }

    match cur {
        Some((_, cur_group)) => {
            acc.push(cur_group);
        }
        None => {}
    }

    Ok(acc)
}

// TODO: we don't need to hash here, this could take a PdObj projection, but I'm too lazy
pub fn identical_by<I, K: Eq, E, F>(it: impl IntoIterator<Item=I>, mut proj: F) -> Result<bool, E> where F: FnMut(&I) -> Result<K, E> {
    let mut acc: Option<K> = None;
    for obj in it {
        match &acc {
            None => { acc = Some(proj(&obj)?); }
            Some(k) => {
                if k != &proj(&obj)? { return Ok(false) }
            }
        }
    }
    Ok(true)
}

pub fn unique_by<I, K: Hash + Eq, E, F>(it: impl IntoIterator<Item=I>, mut proj: F) -> Result<bool, E> where F: FnMut(&I) -> Result<K, E> {
    let mut acc = HashSet::new();
    for obj in it {
        let key = proj(&obj)?;
        if acc.contains(&key) {
            return Ok(false);
        }
        acc.insert(key);
    }
    Ok(true)
}

pub fn uniquify_by<I, K: Hash + Eq, E, F>(it: impl IntoIterator<Item=I>, mut proj: F) -> Result<Vec<I>, E> where F: FnMut(&I) -> Result<K, E> {
    let mut acc = Vec::new();
    let mut seen = HashSet::new();
    for obj in it {
        let key = proj(&obj)?;
        if !seen.contains(&key) {
            acc.push(obj);
        }
        seen.insert(key);
    }
    Ok(acc)
}


pub fn first_duplicate_by<I, K: Hash + Eq, E, F>(it: impl IntoIterator<Item=I>, mut proj: F) -> Result<Option<I>, E> where F: FnMut(&I) -> Result<K, E> {
    let mut acc = HashSet::new();
    for obj in it {
        let key = proj(&obj)?;
        if acc.contains(&key) {
            return Ok(Some(obj));
        }
        acc.insert(key);
    }
    Ok(None)
}
