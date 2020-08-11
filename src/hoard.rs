use std::collections::{VecDeque, HashMap};
use std::slice::Iter;
use std::hash::Hash;
use crate::pderror::{PdError, PdResult, PdUnit};

#[derive(Debug, Clone)]
pub enum Hoard<K,V> {
    Vec(Vec<V>),
    Deque(VecDeque<V>),
    Map(HashMap<K,V>),
}

pub trait HoardKey: Hash + Eq {
    fn to_isize(&self) -> Option<isize>;
    fn from_usize(i: usize) -> Self;

    fn to_pythonic_index(&self, len: usize) -> Option<usize> {
        let mut i = self.to_isize()?;
        if 0 <= i {
            let u = i as usize;
            if u < len { Some(u) } else { None }
        } else {
            i += len as isize;
            if 0 <= i { Some(i as usize) } else { None }
        }
    }
}

impl<K,V> Hoard<K,V> {
    pub fn new() -> Hoard<K,V> { Hoard::Vec(Vec::new()) }

    pub fn push(&mut self, obj: V) -> PdUnit {
        match self {
            Hoard::Vec(a) => a.push(obj),
            Hoard::Deque(a) => a.push_back(obj),
            Hoard::Map(_) => return Err(PdError::InvalidHoardOperation),
        }
        Ok(())
    }

    pub fn push_front(&mut self, obj: V) -> PdUnit {
        match self {
            Hoard::Vec(a) => {
                let mut d = a.drain(..).collect::<VecDeque<V>>();
                d.push_front(obj);
                *self = Hoard::Deque(d);
            }
            Hoard::Deque(a) => a.push_front(obj),
            Hoard::Map(_) => return Err(PdError::InvalidHoardOperation),
        }
        Ok(())
    }

    pub fn extend(&mut self, obj: impl Iterator<Item=V>) -> PdUnit {
        match self {
            Hoard::Vec(a) => a.extend(obj),
            Hoard::Deque(a) => a.extend(obj),
            Hoard::Map(_) => return Err(PdError::InvalidHoardOperation),
        }
        Ok(())
    }

    pub fn pop(&mut self) -> PdResult<Option<V>> {
        match self {
            Hoard::Vec(a) => Ok(a.pop()),
            Hoard::Deque(a) => Ok(a.pop_back()),
            Hoard::Map(_) => Err(PdError::InvalidHoardOperation),
        }
    }

    pub fn pop_front(&mut self) -> PdResult<Option<V>> {
        match self {
            Hoard::Vec(a) => {
                let mut d = a.drain(..).collect::<VecDeque<V>>();
                let ret = d.pop_front();
                *self = Hoard::Deque(d);
                Ok(ret)
            }
            Hoard::Deque(a) => Ok(a.pop_front()),
            Hoard::Map(_) => Err(PdError::InvalidHoardOperation),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Hoard::Vec(a) => a.is_empty(),
            Hoard::Deque(a) => a.is_empty(),
            Hoard::Map(a) => a.is_empty(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Hoard::Vec(a) => a.len(),
            Hoard::Deque(a) => a.len(),
            Hoard::Map(a) => a.len(),
        }
    }
}

impl<K: HoardKey, V> Hoard<K, V> {
    pub fn get(&self, key: &K) -> Option<&V> {
        match self {
            Hoard::Vec(a) => {
                let len = a.len();
                a.get(key.to_pythonic_index(len)?)
            },
            Hoard::Deque(a) => a.get(key.to_pythonic_index(a.len())?),
            Hoard::Map(a) => a.get(key),
        }
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        match self {
            Hoard::Vec(a) => {
                let len = a.len();
                a.get_mut(key.to_pythonic_index(len)?)
            },
            Hoard::Deque(a) => a.get_mut(key.to_pythonic_index(a.len())?),
            Hoard::Map(a) => a.get_mut(key),
        }
    }

    fn force_map(&mut self) -> &mut HashMap<K, V> {
        match self {
            Hoard::Vec(a) => {
                *self = Hoard::Map(a.drain(..).enumerate().map(|(i, e)| (K::from_usize(i), e)).collect());
                self.force_map()
            },
            Hoard::Deque(a) => {
                *self = Hoard::Map(a.drain(..).enumerate().map(|(i, e)| (K::from_usize(i), e)).collect());
                self.force_map()
            },
            Hoard::Map(a) => a,
        }
    }

    pub fn update(&mut self, key: K, value: V) {
        match self.get_mut(&key) {
            Some(pos) => { *pos = value; }
            None => {
                self.force_map().insert(key, value);
            }
        }
    }

    pub fn delete(&mut self, key: &K) {
        self.force_map().remove(&key);
    }

    pub fn replace_vec(&mut self, v: Vec<V>) {
        *self = Hoard::Vec(v);
    }
}

impl<K: Ord, V: Clone> Hoard<K,V> {
    pub fn first(&self) -> Option<&V> {
        match self {
            Hoard::Vec(a) => a.first(),
            Hoard::Deque(a) => a.front(),
            Hoard::Map(a) => {
                a.iter().min_by(|a, b| a.0.cmp(b.0)).map(|x| x.1)
            }
        }
    }

    pub fn last(&self) -> Option<&V> {
        match self {
            Hoard::Vec(a) => a.last(),
            Hoard::Deque(a) => a.back(),
            Hoard::Map(a) => {
                a.iter().max_by(|a, b| a.0.cmp(b.0)).map(|x| x.1)
            }
        }
    }

    pub fn iter(&self) -> HoardIter<'_, V> {
        match self {
            Hoard::Vec(a) => HoardIter::VecIter(a.iter()),
            Hoard::Deque(a) => HoardIter::DequeIter(a.iter()),
            Hoard::Map(a) => {
                let mut items = a.iter().collect::<Vec<(&K, &V)>>();
                items.sort_by(|x, y| x.0.cmp(&y.0));
                HoardIter::MapIter(items.iter().map(|(_, v)| *v).collect(), 0)
            }
        }
    }
}

// TODO: sorting is a little excessive. Might break into iter and unordered_iter
#[derive(Clone)]
pub enum HoardIter<'a, V> {
    VecIter(Iter<'a, V>),
    DequeIter(std::collections::vec_deque::Iter<'a, V>),
    MapIter(Vec<&'a V>, usize),
}

impl<'a, V> Iterator for HoardIter<'a, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<&'a V> {
        match self {
            HoardIter::VecIter(it) => it.next(),
            HoardIter::DequeIter(it) => it.next(),
            HoardIter::MapIter(vec, i) => match vec.get(*i) {
                Some(v) => { *i += 1; Some(*v) }
                None => None
            }
        }
    }
}
