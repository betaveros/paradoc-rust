use std::collections::{VecDeque, HashMap};
use std::slice::Iter;
use crate::pderror::{PdError, PdResult, PdUnit};

#[derive(Debug)]
pub enum Hoard<K,V> {
    Vec(Vec<V>),
    Deque(VecDeque<V>),
    Map(HashMap<K,V>),
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
}

impl<K,V> Hoard<K,V> {
    pub fn is_empty(&self) -> bool {
        match self {
            Hoard::Vec(a) => a.is_empty(),
            Hoard::Deque(a) => a.is_empty(),
            Hoard::Map(a) => a.is_empty(),
        }
    }
}

impl<K: Ord, V: Clone> Hoard<K,V> {
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
