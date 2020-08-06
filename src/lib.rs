#[macro_use] extern crate lazy_static;

use std::cmp::Ordering;
use std::ops::BitAnd;
use std::rc::Rc;
use std::slice::Iter;
use std::fmt::Debug;
use std::mem;
use num_iter;
use num::bigint::BigInt;
use num_traits::cast::ToPrimitive;
use num_traits::pow::Pow;
use std::collections::{HashSet, HashMap};

mod lex;
mod pdnum;
mod pderror;
mod input;
mod slice_util;
use crate::pdnum::{PdNum, PdTotalNum};
use crate::pderror::{PdError, PdResult, PdUnit};
use crate::input::{InputTrigger, ReadValue, EOFReader};

#[derive(Debug)]
pub struct Environment {
    stack: Vec<Rc<PdObj>>,
    x_stack: Vec<Rc<PdObj>>,
    variables: HashMap<String, Rc<PdObj>>,
    marker_stack: Vec<usize>,
    // hmmm...
    shadow: Option<ShadowState>,
    input_trigger: Option<InputTrigger>,
    reader: Option<EOFReader>,
}

#[derive(Debug, Clone, Copy)]
pub enum ShadowType {
    Normal,
    Keep,
}

#[derive(Debug)]
pub struct ShadowState {
    env: Box<Environment>,
    arity: usize,
    shadow_type: ShadowType,
}

impl ShadowState {
    fn pop(&mut self) -> Option<Rc<PdObj>> {
        match self.shadow_type {
            ShadowType::Normal => {
                let res = self.env.pop();
                if res.is_some() { self.arity += 1; }
                res
            }
            ShadowType::Keep => {
                let res = self.env.stack.get(self.env.stack.len() - 1 - self.arity);
                if res.is_some() { self.arity += 1; }
                res.map(Rc::clone)
            }
        }
    }
}

impl Environment {
    fn push(&mut self, obj: Rc<PdObj>) {
        self.stack.push(obj)
    }
    // idk what the idiomatic way is yet
    // fn extend(&mut self, objs: Vec<Rc<PdObj>>) {
    //     for obj in objs {
    //         self.push(obj)
    //     }
    // }
    fn extend(&mut self, objs: Vec<Rc<PdObj>>) {
        for obj in objs {
            self.push(obj)
        }
    }
    fn extend_clone(&mut self, objs: &Vec<Rc<PdObj>>) {
        for obj in objs {
            self.push(Rc::clone(obj))
        }
    }

    fn run_stack_trigger(&mut self) -> Option<Rc<PdObj>> {
        match (&mut self.reader, self.input_trigger) {
            (Some(r), Some(t)) => {
                r.read(t).expect("io error in input trigger").map(|v| Rc::new(PdObj::from(v)))
            }
            _ => None
        }
    }

    fn pop(&mut self) -> Option<Rc<PdObj>> {
        let len = self.stack.len();
        let ret = self.stack.pop();
        match ret {
            Some(v) => {
                for i in (0..self.marker_stack.len()).rev() {
                    if self.marker_stack[i] > len {
                        self.marker_stack[i] = len
                    } else {
                        break
                    }
                }
                Some(v)
            }
            None => match &mut self.shadow {
                Some(inner) => inner.pop(),
                None => self.run_stack_trigger()
            }
        }
    }
    fn pop_result(&mut self, err_msg: &str) -> PdResult<Rc<PdObj>> {
        self.pop().ok_or(PdError::EmptyStack(err_msg.to_string()))
    }
    fn pop_n_result(&mut self, n: usize, err_msg: &str) -> Result<Vec<Rc<PdObj>>, PdError> {
        let mut ret: Vec<Rc<PdObj>> = Vec::new();
        for _ in 0..n {
            ret.push(self.pop_result(err_msg)?);
        }
        ret.reverse();
        Ok(ret)
    }

    fn peek_result(&mut self, err_msg: &str) -> PdResult<Rc<PdObj>> {
        let ret = self.pop_result(err_msg)?;
        self.push(Rc::clone(&ret));
        Ok(ret)
    }

    fn take_stack(&mut self) -> Vec<Rc<PdObj>> {
        mem::take(&mut self.stack)
    }

    fn mark_stack(&mut self) {
        self.marker_stack.push(self.stack.len())
    }

    fn pop_stack_marker(&mut self) -> Option<usize> {
        self.marker_stack.pop()
    }

    fn maximize_length(&mut self) {
        let mut acc = Vec::new();
        while let Some(v) = self.run_stack_trigger() {
            acc.push(v);
        }
        acc.reverse();
        acc.extend(self.stack.drain(..));
        self.stack = acc;
    }

    fn pop_until_stack_marker(&mut self) -> Vec<Rc<PdObj>> {
        match self.pop_stack_marker() {
            Some(marker) => {
                self.stack.split_off(marker) // this is way too perfect
            }
            None => {
                self.maximize_length();
                self.stack.split_off(0) // meh
            }
        }
    }

    fn push_x(&mut self, obj: Rc<PdObj>) {
        self.x_stack.push(obj)
    }

    fn push_yx(&mut self) {
        self.push_x(Rc::new(PdObj::from("INTERNAL Y FILLER -- YOU SHOULD NOT SEE THIS".to_string())));
        self.push_x(Rc::new(PdObj::from("INTERNAL X FILLER -- YOU SHOULD NOT SEE THIS".to_string())));
    }
    fn set_yx(&mut self, y: Rc<PdObj>, x: Rc<PdObj>) {
        let len = self.x_stack.len();
        self.x_stack[len - 2] = y;
        self.x_stack[len - 1] = x;
    }
    fn pop_yx(&mut self) {
        self.x_stack.pop();
        self.x_stack.pop();
    }

    fn short_insert(&mut self, name: &str, obj: PdObj) {
        self.variables.insert(name.to_string(), Rc::new(obj));
    }

    fn get(&self, name: &str) -> Option<&Rc<PdObj>> {
        match &self.shadow {
            Some(inner) => inner.env.get(name),
            None => {
                match name {
                    "X" => self.x_stack.get(self.x_stack.len() - 1),
                    "Y" => self.x_stack.get(self.x_stack.len() - 2),
                    "Z" => self.x_stack.get(self.x_stack.len() - 3),
                    _ => self.variables.get(name)
                }
            }
        }
    }

    pub fn new_with_stdin() -> Environment {
        Environment {
            stack: Vec::new(),
            x_stack: Vec::new(),
            variables: HashMap::new(),
            marker_stack: Vec::new(),
            shadow: None,
            input_trigger: None,
            reader: Some(EOFReader::new(Box::new(std::io::BufReader::new(std::io::stdin())))),
        }
    }

    pub fn new() -> Environment {
        Environment {
            stack: Vec::new(),
            x_stack: Vec::new(),
            variables: HashMap::new(),
            marker_stack: Vec::new(),
            shadow: None,
            input_trigger: None,
            reader: None,
        }
    }

    fn to_string(&self, obj: &Rc<PdObj>) -> String {
        match &**obj {
            PdObj::Num(n) => n.to_string(),
            PdObj::String(s) => s.iter().collect::<String>(),
            PdObj::List(v) => v.iter().map(|o| self.to_string(o)).collect::<Vec<String>>().join(""),
            PdObj::Block(b) => b.code_repr(),
        }
    }

    fn to_repr_string(&self, obj: &Rc<PdObj>) -> String {
        match &**obj {
            PdObj::Num(n) => n.to_string(),
            PdObj::String(s) => format!("\"{}\"", &s.iter().collect::<String>()),
            PdObj::List(v) => format!("[{}]", v.iter().map(|o| self.to_repr_string(o)).collect::<Vec<String>>().join(" ")),
            PdObj::Block(b) => b.code_repr(),
        }
    }

    pub fn stack_to_string(&self) -> String {
        self.stack.iter().map(|x| self.to_string(x) ).collect::<Vec<String>>().join("")
    }
    pub fn stack_to_repr_string(&self) -> String {
        self.stack.iter().map(|x| self.to_repr_string(x) ).collect::<Vec<String>>().join(" ")
    }

    fn run_on_bracketed_shadow<T>(&mut self, shadow_type: ShadowType, body: impl FnOnce(&mut Environment) -> Result<T, PdError>) -> Result<T, PdError> {
        let (ret, _arity) = self.run_on_bracketed_shadow_with_arity(shadow_type, body)?;
        Ok(ret)
    }

    fn run_on_bracketed_shadow_with_arity<T>(&mut self, shadow_type: ShadowType, body: impl FnOnce(&mut Environment) -> Result<T, PdError>) -> Result<(T, usize), PdError> {
        let env = mem::replace(self, Environment::new());

        let mut benv = Environment {
            stack: Vec::new(),
            x_stack: Vec::new(),
            variables: HashMap::new(),
            marker_stack: vec![0],
            shadow: Some(ShadowState { env: Box::new(env), arity: 0, shadow_type }),
            input_trigger: None,
            reader: None,
        };

        let ret = body(&mut benv)?;

        let shadow = benv.shadow.expect("Bracketed shadow disappeared!?!?");
        let arity = shadow.arity;
        *self = *(shadow.env);

        Ok((ret, arity))
    }
}

pub trait Block : Debug {
    fn run(&self, env: &mut Environment) -> PdUnit;
    fn code_repr(&self) -> String;
}

fn sandbox(env: &mut Environment, func: &Rc<dyn Block>, args: Vec<Rc<PdObj>>) -> Result<Vec<Rc<PdObj>>, PdError> {
    env.run_on_bracketed_shadow(ShadowType::Normal, |inner| {
        inner.extend(args);
        func.run(inner)?;
        Ok(inner.take_stack())
    })
}
#[derive(Debug)]
pub enum PdObj {
    Num(Rc<PdNum>),
    String(Rc<Vec<char>>),
    List(Rc<Vec<Rc<PdObj>>>),
    Block(Rc<dyn Block>),
}

impl PartialEq for PdObj {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PdObj::Num   (a), PdObj::Num   (b)) => a == b,
            (PdObj::String(a), PdObj::String(b)) => a == b,
            (PdObj::List  (a), PdObj::List  (b)) => a == b,
            _ => false,
        }
    }
}

impl PartialOrd for PdObj {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (PdObj::Num   (a), PdObj::Num   (b)) => a.partial_cmp(b),
            (PdObj::String(a), PdObj::String(b)) => Some(a.cmp(b)),
            (PdObj::List  (a), PdObj::List  (b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

impl From<PdNum> for PdObj {
    fn from(s: PdNum) -> Self {
        PdObj::Num(Rc::new(s))
    }
}

macro_rules! forward_from {
    ($ty:ident) => {
        impl From<$ty> for PdObj {
            fn from(n: $ty) -> Self { PdObj::Num(Rc::new(PdNum::from(n))) }
        }
    }
}

forward_from!(BigInt);
forward_from!(char);
forward_from!(i32);
forward_from!(f64);
forward_from!(usize);

impl From<Vec<char>> for PdObj {
    fn from(s: Vec<char>) -> Self {
        PdObj::String(Rc::new(s))
    }
}
impl From<String> for PdObj {
    fn from(s: String) -> Self {
        PdObj::String(Rc::new(s.chars().collect()))
    }
}
impl From<&String> for PdObj {
    fn from(s: &String) -> Self {
        PdObj::String(Rc::new(s.chars().collect()))
    }
}
impl From<ReadValue> for PdObj {
    fn from(v: ReadValue) -> Self {
        match v {
            ReadValue::String(s) => PdObj::from(s),
            ReadValue::Char(c) => PdObj::from(c),
            ReadValue::Int(n) => PdObj::from(n),
            ReadValue::Float(f) => PdObj::from(f),
            ReadValue::List(v) => PdObj::List(Rc::new(v.into_iter().map(|x| Rc::new(PdObj::from(x))).collect())),
        }
    }
}

struct BuiltIn {
    name: String,
    func: fn(&mut Environment) -> PdUnit,
}

impl Debug for BuiltIn {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(fmt, "BuiltIn {{ name: {:?}, func: ??? }}", self.name)
    }
}
impl Block for BuiltIn {
    fn run(&self, env: &mut Environment) -> PdUnit {
        (self.func)(env)
    }
    fn code_repr(&self) -> String {
        self.name.clone()
    }
}

// Yeah, so we do want to take args because of interactions with shadow stacks
trait Case {
    fn arity(&self) -> usize;
    fn maybe_run_noncommutatively(&self, env: &mut Environment, args: &Vec<Rc<PdObj>>) -> PdResult<Option<Vec<Rc<PdObj>>>>;
}

struct UnaryAnyCase {
    func: fn(&mut Environment, &Rc<PdObj>) -> PdResult<Vec<Rc<PdObj>>>,
}
impl Case for UnaryAnyCase {
    fn arity(&self) -> usize { 1 }
    fn maybe_run_noncommutatively(&self, env: &mut Environment, args: &Vec<Rc<PdObj>>) -> PdResult<Option<Vec<Rc<PdObj>>>> {
        Ok(Some((self.func)(env, &args[0])?))
    }
}

struct BinaryAnyCase {
    func: fn(&mut Environment, &Rc<PdObj>, &Rc<PdObj>) -> PdResult<Vec<Rc<PdObj>>>,
}
impl Case for BinaryAnyCase {
    fn arity(&self) -> usize { 2 }
    fn maybe_run_noncommutatively(&self, env: &mut Environment, args: &Vec<Rc<PdObj>>) -> PdResult<Option<Vec<Rc<PdObj>>>> {
        Ok(Some((self.func)(env, &args[0], &args[1])?))
    }
}

struct TernaryAnyCase {
    func: fn(&mut Environment, &Rc<PdObj>, &Rc<PdObj>, &Rc<PdObj>) -> PdResult<Vec<Rc<PdObj>>>,
}
impl Case for TernaryAnyCase {
    fn arity(&self) -> usize { 3 }
    fn maybe_run_noncommutatively(&self, env: &mut Environment, args: &Vec<Rc<PdObj>>) -> PdResult<Option<Vec<Rc<PdObj>>>> {
        Ok(Some((self.func)(env, &args[0], &args[1], &args[2])?))
    }
}

fn just_num(obj: &PdObj) -> Option<Rc<PdNum>> {
    match obj {
        PdObj::Num(n) => Some(Rc::clone(n)),
        _ => None,
    }
}
// TODO these should really PdError when they get a PdNum that fails
// or maybe clamp or something, if we always use them that way? idk
fn num_to_isize(obj: &PdObj) -> Option<isize> {
    match obj {
        PdObj::Num(n) => n.to_isize(),
        _ => None,
    }
}
fn num_to_nn_usize(obj: &PdObj) -> Option<usize> {
    match obj {
        PdObj::Num(n) => n.to_nn_usize(),
        _ => None,
    }
}
fn just_string(obj: &PdObj) -> Option<Rc<Vec<char>>> {
    match obj {
        PdObj::String(s) => Some(Rc::clone(s)),
        _ => None,
    }
}

pub enum PdSeq {
    List(Rc<Vec<Rc<PdObj>>>),
    String(Rc<Vec<char>>),
    Range(BigInt, BigInt),
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum PdSeqBuildType { String, NotString }

impl BitAnd for PdSeqBuildType {
    type Output = PdSeqBuildType;

    fn bitand(self, other: PdSeqBuildType) -> PdSeqBuildType {
        match (self, other) {
            (PdSeqBuildType::String,    PdSeqBuildType::String) =>    PdSeqBuildType::String,
            (PdSeqBuildType::String,    PdSeqBuildType::NotString) => PdSeqBuildType::NotString,
            (PdSeqBuildType::NotString, PdSeqBuildType::String) =>    PdSeqBuildType::NotString,
            (PdSeqBuildType::NotString, PdSeqBuildType::NotString) => PdSeqBuildType::NotString,
        }
    }
}

impl PdSeq {
    fn build_type(&self) -> PdSeqBuildType {
        match self {
            PdSeq::List(_) => PdSeqBuildType::NotString,
            PdSeq::String(_) => PdSeqBuildType::String,
            PdSeq::Range(_, _) => PdSeqBuildType::NotString,
        }
    }

    fn to_rc_pd_obj(self) -> Rc<PdObj> {
        match self {
            PdSeq::List(rc) => Rc::new(PdObj::List(rc)),
            PdSeq::String(s) => Rc::new(PdObj::String(s)),
            PdSeq::Range(_, _) => Rc::new(PdObj::List(Rc::new(self.to_new_vec()))),
            // PdSeq::Range(a, b) => num_iter::range(BigInt::clone(a), BigInt::clone(b)).map(|x| Rc::new(PdObj::from(x))).collect(),
        }
    }
}

pub enum PdSeqElement {
    ListElement(Rc<PdObj>),
    Char(char),
    Int(BigInt),
}

impl PdSeqElement {
    fn to_rc_pd_obj(self) -> Rc<PdObj> {
        match self {
            PdSeqElement::ListElement(rc) => rc,
            PdSeqElement::Char(c) => Rc::new(PdObj::from(c)),
            PdSeqElement::Int(n) => Rc::new(PdObj::from(n)),
        }
    }
}

pub enum PdIter<'a> {
    List(Iter<'a, Rc<PdObj>>),
    String(Iter<'a, char>), // String(std::str::Chars<'a>),
    Range(num_iter::Range<BigInt>),
}

impl PdSeq {
    fn iter(&self) -> PdIter<'_> {
        match self {
            PdSeq::List(v) => PdIter::List(v.iter()),
            PdSeq::String(s) => PdIter::String(s.iter()),
            PdSeq::Range(a, b) => PdIter::Range(num_iter::range(BigInt::clone(a), BigInt::clone(b))),
        }
    }

    fn len(&self) -> usize {
        match self {
            PdSeq::List(v) => v.len(),
            PdSeq::String(s) => s.len(),
            PdSeq::Range(a, b) => (b - a).to_usize().expect("obnoxious bigint range"),
        }
    }

    fn index(&self, i: usize) -> Option<PdSeqElement> {
        match self {
            PdSeq::List(v) => v.get(i).map(|e| PdSeqElement::ListElement(Rc::clone(e))),
            PdSeq::String(s) => s.get(i).map(|e| PdSeqElement::Char(*e)),
            PdSeq::Range(a, b) => {
                let guess = a + i;
                if &guess < b {
                    Some(PdSeqElement::Int(guess))
                } else {
                    None
                }
            }
        }
    }

    fn pythonic_index(&self, i: isize) -> Option<PdSeqElement> {
        if 0 <= i {
            self.index(i as usize)
        } else {
            self.index((self.len() as isize + i) as usize)
        }
    }

    fn pythonic_clamp_slice_index(&self, index: isize) -> usize {
        let len = self.len();
        if 0 <= index {
            (index as usize).min(len)
        } else {
            let min_index = -(len as isize);
            (len as isize + index.max(min_index)) as usize
        }
    }

    fn pythonic_split_left(&self, index: isize) -> PdSeq {
        let uindex = self.pythonic_clamp_slice_index(index);

        match self {
            PdSeq::List(v) => PdSeq::List(Rc::new(v.split_at(uindex).0.to_vec())),
            PdSeq::String(s) => PdSeq::String(Rc::new(s.split_at(uindex).0.to_vec())),
            PdSeq::Range(a, b) => PdSeq::Range(BigInt::clone(a), BigInt::clone(b.min(&(a + uindex)))),
            //, PdSeq::Range(a + uindex, BigInt::clone(b))),
        }
    }

    fn pythonic_mod_slice(&self, modulus: isize) -> PdResult<PdSeq> {
        match self {
            PdSeq::List(v) => Ok(PdSeq::List(Rc::new(slice_util::pythonic_mod_slice(&**v, modulus).ok_or(PdError::IndexError("mod slice sad".to_string()))?.iter().cloned().cloned().collect()))),
            PdSeq::String(s) => Ok(PdSeq::String(Rc::new(slice_util::pythonic_mod_slice(&**s, modulus).ok_or(PdError::IndexError("mod slice sad".to_string()))?.iter().cloned().cloned().collect()))),
            PdSeq::Range(_, _) => Ok(PdSeq::List(Rc::new(slice_util::pythonic_mod_slice(&self.to_new_vec(), modulus).ok_or(PdError::IndexError("mod slice sad".to_string()))?.iter().cloned().cloned().collect()))),
        }
    }

    fn rev_copy(&self) -> PdSeq {
        match self {
            PdSeq::List(v)     => PdSeq::List  (Rc::new(slice_util::rev_copy(&**v).iter().cloned().cloned().collect())),
            PdSeq::String(s)   => PdSeq::String(Rc::new(slice_util::rev_copy(&**s).iter().cloned().cloned().collect())),
            PdSeq::Range(_, _) => PdSeq::List  (Rc::new(slice_util::rev_copy(&self.to_new_vec()).iter().cloned().cloned().collect())),
        }
    }

    fn pythonic_split_right(&self, index: isize) -> PdSeq {
        let uindex = self.pythonic_clamp_slice_index(index);

        match self {
            PdSeq::List(v) => PdSeq::List(Rc::new(v.split_at(uindex).1.to_vec())),
            PdSeq::String(s) => PdSeq::String(Rc::new(s.split_at(uindex).1.to_vec())),
            PdSeq::Range(a, b) => PdSeq::Range(a + uindex, BigInt::clone(b)),
        }
    }

    // TODO: expensive idk scary
    // it's probably fine but want to make sure it's necessary and I don't accidentally compose it
    // with additional clones
    fn to_new_vec(&self) -> Vec<Rc<PdObj>> {
        match self {
            PdSeq::List(v) => (&**v).clone(),
            PdSeq::String(s) => s.iter().map(|x| Rc::new(PdObj::from(*x))).collect(),
            PdSeq::Range(a, b) => num_iter::range(BigInt::clone(a), BigInt::clone(b)).map(|x| Rc::new(PdObj::from(x))).collect(),
        }
    }

    fn first(&self) -> Option<PdSeqElement> {
        match self {
            PdSeq::List(v) => Some(PdSeqElement::ListElement(Rc::clone(v.first()?))),
            PdSeq::String(s) => Some(PdSeqElement::Char(*s.first()?)),
            PdSeq::Range(a, b) => if a < b { Some(PdSeqElement::Int(BigInt::clone(a))) } else { None },
        }
    }

    fn split_first(&self) -> Option<(Rc<PdObj>, Rc<PdObj>)> {
        match self {
            PdSeq::List(v) => {
                let (x, xs) = v.split_first()?;
                Some((Rc::clone(x), pd_list(xs.to_vec())))
            }
            PdSeq::String(s) => {
                let (x, xs) = s.split_first()?;
                Some((Rc::new(PdObj::from(*x)), Rc::new(PdObj::String(Rc::new(xs.to_vec())))))
            }
            PdSeq::Range(_, _) => {
                let v = self.to_new_vec();
                let (x, xs) = v.split_first()?;
                Some((Rc::clone(x), pd_list(xs.to_vec())))
            }
        }
    }

    fn last(&self) -> Option<PdSeqElement> {
        match self {
            PdSeq::List(v) => Some(PdSeqElement::ListElement(Rc::clone(v.last()?))),
            PdSeq::String(s) => Some(PdSeqElement::Char(*s.last()?)),
            PdSeq::Range(a, b) => if a < b { Some(PdSeqElement::Int(b - 1)) } else { None },
        }
    }

    fn split_last(&self) -> Option<(Rc<PdObj>, Rc<PdObj>)> {
        match self {
            PdSeq::List(v) => {
                let (x, xs) = v.split_last()?;
                Some((Rc::clone(x), pd_list(xs.to_vec())))
            }
            PdSeq::String(s) => {
                let (x, xs) = s.split_last()?;
                Some((Rc::new(PdObj::from(*x)), Rc::new(PdObj::String(Rc::new(xs.to_vec())))))
            }
            PdSeq::Range(_, _) => {
                let v = self.to_new_vec();
                let (x, xs) = v.split_last()?;
                Some((Rc::clone(x), pd_list(xs.to_vec())))
            }
        }
    }
}

impl Iterator for PdIter<'_> {
    type Item = Rc<PdObj>;

    fn next(&mut self) -> Option<Rc<PdObj>> {
        match self {
            PdIter::List(it) => it.next().map(Rc::clone),
            PdIter::String(cs) => cs.next().map(|x| Rc::new(PdObj::from(*x))),
            PdIter::Range(rs) => rs.next().map(|x| Rc::new(PdObj::from(x))),
        }
    }
}

fn just_seq(obj: &PdObj) -> Option<PdSeq> {
    match obj {
        PdObj::List(a) => Some(PdSeq::List(Rc::clone(a))),
        PdObj::String(a) => Some(PdSeq::String(Rc::clone(a))),
        _ => None,
    }
}
// TODO: hrmm do we want to make this take Rc<PdObj>
// honestly we should just get rid of that extra Rc, it's pretty wasteful
fn list_singleton(obj: &PdObj) -> Option<Rc<Vec<Rc<PdObj>>>> {
    match obj {
        PdObj::Num(n) => Some(Rc::new(vec![Rc::new(PdObj::from(PdNum::clone(&**n)))])),
        PdObj::List(x) => Some(Rc::clone(x)),
        _ => None,
    }
}
fn seq_range(obj: &PdObj) -> Option<PdSeq> {
    match obj {
        PdObj::Num(num) => match &**num {
            PdNum::Int(a) => Some(PdSeq::Range(BigInt::from(0), BigInt::clone(a))),
            _ => None,
        },
        _ => just_seq(obj),
    }
}
fn just_block(obj: &PdObj) -> Option<Rc<dyn Block>> {
    match obj {
        PdObj::Block(a) => Some(Rc::clone(a)),
        _ => None,
    }
}

struct UnaryNumCase {
    func: fn(&mut Environment, &PdNum) -> PdResult<Vec<Rc<PdObj>>>,
}
impl Case for UnaryNumCase {
    fn arity(&self) -> usize { 1 }
    fn maybe_run_noncommutatively(&self, env: &mut Environment, args: &Vec<Rc<PdObj>>) -> PdResult<Option<Vec<Rc<PdObj>>>> {
        match &*args[0] {
            PdObj::Num(ai) => {
                let ai2 = Rc::clone(ai);
                Ok(Some((self.func)(env, &*ai2)?))
            }
            _ => Ok(None)
        }
    }
}

struct UnaryCase<T> {
    coerce: fn(&PdObj) -> Option<T>,
    func: fn(&mut Environment, &T) -> PdResult<Vec<Rc<PdObj>>>,
}
impl<T> Case for UnaryCase<T> {
    fn arity(&self) -> usize { 1 }
    fn maybe_run_noncommutatively(&self, env: &mut Environment, args: &Vec<Rc<PdObj>>) -> PdResult<Option<Vec<Rc<PdObj>>>> {
        match (self.coerce)(&*args[0]) {
            Some(a) => {
                Ok(Some((self.func)(env, &a)?))
            }
            _ => Ok(None)
        }
    }
}
fn unary_num_case(func: fn(&mut Environment, &Rc<PdNum>) -> PdResult<Vec<Rc<PdObj>>>) -> Rc<dyn Case> {
    Rc::new(UnaryCase { coerce: just_num, func })
}

struct BinaryCase<T1, T2> {
    coerce1: fn(&PdObj) -> Option<T1>,
    coerce2: fn(&PdObj) -> Option<T2>,
    func: fn(&mut Environment, &T1, &T2) -> PdResult<Vec<Rc<PdObj>>>,
}
impl<T1, T2> Case for BinaryCase<T1, T2> {
    fn arity(&self) -> usize { 2 }
    fn maybe_run_noncommutatively(&self, env: &mut Environment, args: &Vec<Rc<PdObj>>) -> PdResult<Option<Vec<Rc<PdObj>>>> {
        match ((self.coerce1)(&*args[0]), (self.coerce2)(&*args[1])) {
            (Some(a), Some(b)) => {
                Ok(Some((self.func)(env, &a, &b)?))
            }
            _ => Ok(None)
        }
    }
}
fn binary_num_case(func: fn(&mut Environment, &Rc<PdNum>, &Rc<PdNum>) -> PdResult<Vec<Rc<PdObj>>>) -> Rc<dyn Case> {
    Rc::new(BinaryCase { coerce1: just_num, coerce2: just_num, func })
}
fn unary_seq_case(func: fn(&mut Environment, &PdSeq) -> PdResult<Vec<Rc<PdObj>>>) -> Rc<dyn Case> {
    Rc::new(UnaryCase { coerce: just_seq, func })
}
fn unary_seq_range_case(func: fn(&mut Environment, &PdSeq) -> PdResult<Vec<Rc<PdObj>>>) -> Rc<dyn Case> {
    Rc::new(UnaryCase { coerce: seq_range, func })
}
struct CasedBuiltIn {
    name: String,
    cases: Vec<Rc<dyn Case>>,
}

impl Debug for CasedBuiltIn {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(fmt, "CasedBuiltIn {{ name: {:?}, cases: [{} ???] }}", self.name, self.cases.len())
    }
}
impl Block for CasedBuiltIn {
    fn run(&self, env: &mut Environment) -> PdUnit {
        let mut done = false;
        let mut accumulated_args: Vec<Rc<PdObj>> = Vec::new();
        for case in &self.cases {
            while accumulated_args.len() < case.arity() {
                match env.pop() {
                    Some(arg) => {
                        accumulated_args.insert(0, arg);
                    }
                    None => {
                        return Err(PdError::EmptyStack(format!("built-in {}", self.name)))
                    }
                }
            }
            match case.maybe_run_noncommutatively(env, &accumulated_args)? {
                Some(res) => {
                    env.stack.extend(res);
                    done = true;
                    break
                }
                None => {
                    if accumulated_args.len() >= 2 {
                        let len = accumulated_args.len();
                        accumulated_args.swap(len - 1, len - 2);
                        match case.maybe_run_noncommutatively(env, &accumulated_args)? {
                            Some(res) => {
                                env.stack.extend(res);
                                done = true;
                                break
                            }
                            None => {}
                        };
                        accumulated_args.swap(len - 1, len - 2);
                    }
                }
            }
        }
        if done {
            Ok(())
        } else {
            Err(PdError::BadArgument(format!("No cases of {} applied!", self.name)))
        }
    }
    fn code_repr(&self) -> String {
        self.name.clone()
    }
}

struct DeepBinaryOpBlock {
    func: fn(&PdNum, &PdNum) -> PdNum,
    other: PdNum,
}
impl Debug for DeepBinaryOpBlock {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(fmt, "DeepBinaryOpBlock {{ func: ???, other: {:?} }}", self.other)
    }
}
impl Block for DeepBinaryOpBlock {
    fn run(&self, env: &mut Environment) -> PdUnit {
        let res = pd_deep_map(&|x| (self.func)(x, &self.other), &*env.pop_result("deep binary op no stack")?);
        env.push(Rc::new(res));
        Ok(())
    }
    fn code_repr(&self) -> String {
        self.other.to_string() + "_???_binary_op"
    }
}

// TODO: handle continue/break (have fun!) (this should be fine now)
// doesn't push and pop yx
// The inner function returns true to break out of the loop.
fn pd_flatmap_foreach_core<F>(env: &mut Environment, func: &Rc<dyn Block>, mut body: F, it: PdIter) -> PdUnit where F: FnMut(Rc<PdObj>) -> PdResult<bool> {
    let mut broken: bool = false;
    for (i, obj) in it.enumerate() {
        env.set_yx(Rc::new(PdObj::from(i)), Rc::clone(&obj));
        for e in sandbox(env, &func, vec![obj])? {
            if body(e)? { broken = true; break; }
        }
        if broken { break; }
    }
    Ok(())
}

fn pd_flatmap_foreach<F>(env: &mut Environment, func: &Rc<dyn Block>, body: F, it: PdIter) -> PdUnit where F: FnMut(Rc<PdObj>) -> PdResult<bool> {
    env.push_yx();
    let core = pd_flatmap_foreach_core(env, func, body, it);
    env.pop_yx();
    core
}

fn pd_map(env: &mut Environment, func: &Rc<dyn Block>, it: PdIter) -> PdResult<Vec<Rc<PdObj>>> {
    let mut acc = Vec::new();
    pd_flatmap_foreach(env, func, |o| { acc.push(o); Ok(false) }, it)?;
    Ok(acc)
}

// TODO: I don't think F should need to borrow &B, but it's tricky.
fn pd_flat_fold_with_short_circuit<B, F>(env: &mut Environment, func: &Rc<dyn Block>, init: B, body: F, it: PdIter) -> PdResult<B> where F: Fn(&B, Rc<PdObj>) -> PdResult<(bool, B)> {
    let mut acc = init;
    pd_flatmap_foreach(env, func, |o| {
        let (do_break, acc2) = body(&acc, o)?;
        acc = acc2;
        Ok(do_break)
    }, it)?;
    Ok(acc)
}

fn pd_flat_fold<B, F>(env: &mut Environment, func: &Rc<dyn Block>, init: B, body: F, it: PdIter) -> PdResult<B> where F: Fn(&B, Rc<PdObj>) -> PdResult<B> {
    let mut acc = init;
    pd_flatmap_foreach(env, func, |o| { acc = body(&acc, o)?; Ok(false) }, it)?;
    Ok(acc)
}

struct OneBodyBlock {
    name: &'static str,
    body: Rc<dyn Block>,
    f: fn(&mut Environment, &Rc<dyn Block>) -> PdUnit,
}
impl Debug for OneBodyBlock {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(fmt, "OneBodyBlock {{ name: {:?}, body: {:?}, f: ??? }}", self.name, self.body)
    }
}
impl Block for OneBodyBlock {
    fn run(&self, env: &mut Environment) -> PdUnit {
        (self.f)(env, &self.body)
    }
    fn code_repr(&self) -> String {
        format!("{}_{}", self.body.code_repr(), self.name)
    }
}
fn pop_seq_range_for(env: &mut Environment, name: &'static str) -> PdResult<PdSeq> {
    let opt_seq = seq_range(&*env.pop_result(format!("{} no stack", name).as_str())?);
    opt_seq.ok_or(PdError::BadArgument(format!("{} coercion fail", name)))
}

#[derive(Debug)]
struct BindBlock {
    body: Rc<dyn Block>,
    bound_object: Rc<PdObj>,
}
impl Block for BindBlock {
    fn run(&self, env: &mut Environment) -> PdUnit {
        env.push(Rc::clone(&self.bound_object));
        self.body.run(env)
    }
    fn code_repr(&self) -> String {
        self.body.code_repr() + "_bind"
    }
}

#[derive(Debug, PartialEq)]
pub enum RcLeader {
    Lit(Rc<PdObj>),
    Var(Rc<String>),
}

#[derive(Debug)]
pub struct CodeBlock(Vec<lex::Trailer>, Vec<RcToken>);

#[derive(Debug, PartialEq)]
pub struct RcToken(pub RcLeader, pub Vec<lex::Trailer>);

fn rcify(tokens: Vec<lex::Token>) -> Vec<RcToken> {
    // for .. in .., which is implicitly into_iter(), can move ownership out of the array
    // (iter() borrows elements only, but we are consuming tokens here)
    tokens.into_iter().map(|lex::Token(leader, trailer)| {
        let rcleader = match leader {
            lex::Leader::StringLit(s) => {
                RcLeader::Lit(Rc::new(PdObj::from(s)))
            }
            lex::Leader::IntLit(n) => {
                RcLeader::Lit(Rc::new(PdObj::from(n)))
            }
            lex::Leader::CharLit(c) => {
                RcLeader::Lit(Rc::new(PdObj::from(c)))
            }
            lex::Leader::FloatLit(f) => {
                RcLeader::Lit(Rc::new(PdObj::from(f)))
            }
            lex::Leader::Block(t, b) => {
                RcLeader::Lit(Rc::new(PdObj::Block(Rc::new(CodeBlock(t, rcify(b))))))
            }
            lex::Leader::Var(s) => {
                RcLeader::Var(Rc::new(s))
            }
        };
        RcToken(rcleader, trailer)
    }).collect()
}

// fn rc_parse_pair((trailers: Vec<lex::Trailer>, tokens

impl CodeBlock {
    pub fn parse(code: &str) -> Self {
        let (init_trailers, tokens) = lex::parse(code);
        let rcs = rcify(tokens);
        CodeBlock(init_trailers, rcs)
    }
}

fn obb(name: &'static str, bb: &Rc<dyn Block>, f: fn(&mut Environment, &Rc<dyn Block>) -> PdUnit) -> PdResult<(Rc<PdObj>, bool)> {
    let body: Rc<dyn Block> = Rc::clone(bb);
    Ok((Rc::new(PdObj::Block(Rc::new(OneBodyBlock { name, body, f }))), false))
}
fn apply_trailer(outer_env: &mut Environment, obj: &Rc<PdObj>, trailer0: &lex::Trailer) -> PdResult<(Rc<PdObj>, bool)> {
    let mut trailer: &str = trailer0.0.as_ref();
    trailer = trailer.strip_prefix('_').unwrap_or(trailer);

    match &**obj {
        PdObj::Num(n) => match trailer {
            "m" | "minus" => { Ok((Rc::new(PdObj::from(-&**n)), false)) }
            "h" | "hundred" => { Ok((Rc::new(PdObj::from(n.mul_const(100))), false)) }
            "k" | "thousand" => { Ok((Rc::new(PdObj::from(n.mul_const(1000))), false)) }
            "p" | "power" => {
                let exponent: PdNum = PdNum::clone(n);
                Ok((Rc::new(PdObj::Block(Rc::new(DeepBinaryOpBlock { func: |a, b| a.pow_num(b), other: exponent }))), false))
            }
            _ => Err(PdError::InapplicableTrailer(format!("{:?} on {:?}", trailer, obj)))
        }
        PdObj::Block(bb) => match trailer {
            "b" | "bind" => {
                let b = outer_env.pop_result("bind nothing to bind")?;
                Ok((Rc::new(PdObj::Block(Rc::new(BindBlock {
                    body: Rc::clone(bb),
                    bound_object: b,
                }))), true))
            }
            "e" | "each" => obb("each", bb, |env, body| {
                let seq = pop_seq_range_for(env, "each")?;
                for obj in seq.iter() {
                    env.push(Rc::clone(&obj));
                    body.run(env)?;
                }
                Ok(())
            }),
            "m" | "map" => obb("map", bb, |env, body| {
                let seq = pop_seq_range_for(env, "map")?;
                let res = pd_map(env, body, seq.iter())?;
                env.push(pd_list(res));
                Ok(())
            }),
            "u" | "under" => obb("under", bb, |env, body| {
                let obj = env.pop_result("under no stack")?;
                body.run(env)?;
                env.push(obj);
                Ok(())
            }),
            "k" | "keep" => obb("keep", bb, |env, body| {
                let res = env.run_on_bracketed_shadow(ShadowType::Keep, |inner| {
                    body.run(inner)?;
                    Ok(inner.take_stack())
                })?;
                env.extend(res);
                Ok(())
            }),
            "q" | "keepunder" => obb("keepunder", bb, |env, body| {
                let (res, arity) = env.run_on_bracketed_shadow_with_arity(ShadowType::Keep, |inner| {
                    body.run(inner)?;
                    Ok(inner.take_stack())
                })?;
                let temp = env.pop_n_result(arity, "keepunder stack failed")?;
                env.extend(res);
                env.extend(temp);
                Ok(())
            }),
            "d" | "double" => obb("double", bb, |env, body| {
                let res = env.run_on_bracketed_shadow(ShadowType::Normal, |inner| {
                    body.run(inner)?;
                    Ok(inner.take_stack())
                })?;
                body.run(env)?;
                env.extend(res);
                Ok(())
            }),
            "š" | "sum" => obb("sum", bb, |env, body| {
                let seq = pop_seq_range_for(env, "sum")?;
                let res = pd_flat_fold(env, body, PdNum::from(0),
                    |acc, o| { Ok(acc + &pd_deep_sum(&o)?) }, seq.iter())?;
                env.push(Rc::new(PdObj::from(res)));
                Ok(())
            }),
            "â" | "all" => obb("all", bb, |env, body| {
                let seq = pop_seq_range_for(env, "all")?;
                let res = pd_flat_fold_with_short_circuit(env, body, true,
                    |_, o| { if pd_truthy(&o) { Ok((true, false)) } else { Ok((false, true)) } }, seq.iter())?;
                env.push(Rc::new(PdObj::from(bi_iverson(res))));
                Ok(())
            }),
            "v" | "bindmap" | "vectorize" => obb("all", bb, |env, body| {
                let b = env.pop_result("bindmap nothing to bind")?;
                let bb: Rc<dyn Block> = Rc::new(BindBlock {
                    body: Rc::clone(body),
                    bound_object: b,
                });

                let seq = pop_seq_range_for(env, "bindmap")?;
                let res = pd_map(env, &bb, seq.iter())?;
                env.push(pd_list(res));
                Ok(())
            }),

            _ => Err(PdError::InapplicableTrailer(format!("{:?} on {:?}", trailer, obj)))
        }
        _ => Err(PdError::InapplicableTrailer(format!("{:?} on {:?}", trailer, obj)))
    }
}

fn apply_all_trailers(env: &mut Environment, mut obj: Rc<PdObj>, mut reluctant: bool, trailer: &[lex::Trailer]) -> PdResult<(Rc<PdObj>, bool)> {
    for t in trailer {
        let np = apply_trailer(env, &obj, t)?; // unwraps or returns Err from entire function
        obj = np.0;
        reluctant = np.1;
    }
    Ok((obj, reluctant))
}

fn lookup_and_break_trailers<'a, 'b>(env: &'a Environment, leader: &str, trailers: &'b[lex::Trailer]) -> Option<(&'a Rc<PdObj>, &'b[lex::Trailer])> {

    let mut var: String = leader.to_string();

    let mut last_found: Option<(&'a Rc<PdObj>, &'b[lex::Trailer])> = None;

    if let Some(res) = env.get(leader) {
        last_found = Some((res, trailers)); // lowest priority
    }

    for (i, t) in trailers.iter().enumerate() {
        var.push_str(&t.0);

        if let Some(res) = env.get(&var) {
            last_found = Some((res, &trailers[i+1..]));
        }
    }

    last_found
}

impl Block for CodeBlock {
    fn run(&self, mut env: &mut Environment) -> PdUnit {
        for init_trailer in &self.0 {
            let mut trailer: &str = init_trailer.0.as_ref();
            trailer = trailer.strip_prefix('_').unwrap_or(trailer);
            match trailer {
                "i" | "input"       => { env.input_trigger = Some(InputTrigger::All       ); }
                "l" | "line"        => { env.input_trigger = Some(InputTrigger::Line      ); }
                "w" | "word"        => { env.input_trigger = Some(InputTrigger::Word      ); }
                "v" | "values"      => { env.input_trigger = Some(InputTrigger::Value     ); }
                "r" | "record"      => { env.input_trigger = Some(InputTrigger::Record    ); }
                "a" | "linearray"   => { env.input_trigger = Some(InputTrigger::AllLines  ); }
                "y" | "valuearray"  => { env.input_trigger = Some(InputTrigger::AllValues ); }
                "q" | "recordarray" => { env.input_trigger = Some(InputTrigger::AllRecords); }
                _ => { panic!("unsupported init trailer"); }
            }
        }

        for RcToken(leader, trailer) in &self.1 {
            // println!("{:?} {:?}", leader, trailer);
            let (obj, reluctant) = match leader {
                RcLeader::Lit(obj) => {
                    apply_all_trailers(env, Rc::clone(obj), true, trailer)?
                }
                RcLeader::Var(s) => {
                    let (obj, rest) = lookup_and_break_trailers(env, s, trailer).ok_or(PdError::UndefinedVariable(String::clone(s)))?;
                    let cobj = Rc::clone(obj); // borrow checker to drop obj which unborrows env
                    apply_all_trailers(env, cobj, false, rest)?
                }
            };

            if reluctant {
                env.push(obj);
            } else {
                apply_on(&mut env, obj)?;
            }
        }
        Ok(())
    }
    fn code_repr(&self) -> String {
        "???".to_string()
    }
}

fn apply_on(env: &mut Environment, obj: Rc<PdObj>) -> PdUnit {
    match &*obj {
        PdObj::Num(_)    => { env.push(obj); Ok(()) }
        PdObj::String(_) => { env.push(obj); Ok(()) }
        PdObj::List(_)   => { env.push(obj); Ok(()) }
        PdObj::Block(bb) => bb.run(env),
    }
}

macro_rules! n_n {
    ($a:ident, $x:expr) => {
        unary_num_case(|_, a| {
            let $a: &PdNum = a;
            Ok(vec![Rc::new(PdObj::from($x))])
        })
    };
}
macro_rules! nn_n {
    ($a:ident, $b:ident, $x:expr) => {
        binary_num_case(|_, a, b| {
            let $a: &PdNum = &**a;
            let $b: &PdNum = &**b;
            Ok(vec![Rc::new(PdObj::from($x))])
        })
    };
}
/*
    let dup_case   : Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| Ok(cc![a, a]) });
    add_cases(":", cc![dup_case]);
*/

// I don't really understand what's going on but apparently $(xyz),* has no trailing commas
// and $(xyz,)* does and maybe the latter is compatible with the former but not vice versa?

macro_rules! cc {
    ($($case:expr),*) => {
        vec![$( Rc::clone(&$case) ),*];
    }
}

macro_rules! juggle {
    ($a:ident -> $($res:expr),*) => {
        { let v: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, $a| Ok(cc![ $( $res ),* ]) }); v }
    };
    ($a:ident, $b:ident -> $($res:expr),*) => {
        { let v: Rc<dyn Case> = Rc::new(BinaryAnyCase { func: |_, $a, $b| Ok(cc![ $( $res ),* ]) }); v }
    };
    ($a:ident, $b:ident, $c:ident -> $($res:expr),*) => {
        { let v: Rc<dyn Case> = Rc::new(TernaryAnyCase { func: |_, $a, $b, $c| Ok(cc![ $( $res ),* ]) }); v }
    };
}

fn pd_list(xs: Vec<Rc<PdObj>>) -> Rc<PdObj> { Rc::new(PdObj::List(Rc::new(xs))) }

fn pd_truthy(x: &PdObj) -> bool {
    match x {
        PdObj::Num(n) => n.is_nonzero(),
        PdObj::String(s) => !s.is_empty(),
        PdObj::List(v) => !v.is_empty(),
        PdObj::Block(_) => true,
    }
}

fn pd_deep_length(x: &PdObj) -> PdResult<usize> {
    match x {
        PdObj::Num(_) => Ok(1),
        PdObj::String(ss) => Ok(ss.len()),
        PdObj::List(v) => v.iter().map(|x| pd_deep_length(&*x)).sum(),
        PdObj::Block(_) => Err(PdError::BadArgument("deep length can't block".to_string())),
    }
}

fn pd_deep_sum(x: &PdObj) -> PdResult<PdNum> {
    match x {
        PdObj::Num(n) => Ok((&**n).clone()),
        PdObj::String(ss) => Ok(PdNum::Char(ss.iter().map(|x| BigInt::from(*x as u32)).sum())),
        PdObj::List(v) => v.iter().map(|x| pd_deep_sum(&*x)).sum(),
        PdObj::Block(_) => Err(PdError::BadArgument("deep sum can't block".to_string())),
    }
}

fn pd_deep_square_sum(x: &PdObj) -> PdResult<PdNum> {
    match x {
        PdObj::Num(n) => Ok(&**n * &**n),
        PdObj::String(ss) => Ok(PdNum::Char(ss.iter().map(|x| BigInt::from(*x as u32).pow(2u32)).sum())),
        PdObj::List(v) => v.iter().map(|x| pd_deep_square_sum(&*x)).sum(),
        PdObj::Block(_) => Err(PdError::BadArgument("deep square sum can't block".to_string())),
    }
}

fn pd_deep_standard_deviation(x: &PdObj) -> PdResult<PdNum> {
    let c = PdNum::from(pd_deep_length(x)?);
    let s = pd_deep_sum(x)?;
    let q = pd_deep_square_sum(x)?;
    ((&(q - s.pow(2u32)) / &c) / (&c - &PdNum::from(1))).sqrt().ok_or(PdError::NumericError("sqrt in stddev failed"))
}

fn pd_deep_product(x: &PdObj) -> PdResult<PdNum> {
    match x {
        PdObj::Num(n) => Ok((&**n).clone()),
        PdObj::String(ss) => Ok(PdNum::Char(ss.iter().map(|x| BigInt::from(*x as u32)).product())),
        PdObj::List(v) => v.iter().map(|x| pd_deep_product(&*x)).product(),
        PdObj::Block(_) => Err(PdError::BadArgument("deep product can't block".to_string())),
    }
}

fn pd_build_if_string(x: Vec<PdNum>) -> PdObj {
    let mut chars: Vec<char> = Vec::new();
    let mut char_ok = true;
    for n in &x {
        match n {
            PdNum::Char(c) => { chars.push(c.to_u32().and_then(std::char::from_u32).expect("char, c'mon")); }
            _ => { char_ok = false; break }
        }
    }
    if char_ok {
        PdObj::from(chars)
    } else {
        PdObj::List(Rc::new(x.iter().map(|e| Rc::new(PdObj::Num(Rc::new(e.clone())))).collect()))
    }
}

fn pd_deep_map<F>(f: &F, x: &PdObj) -> PdObj
    where F: Fn(&PdNum) -> PdNum {

    match x {
        PdObj::Num(n) => PdObj::Num(Rc::new(f(&*n))),
        PdObj::String(s) => pd_build_if_string(s.iter().map(|c| f(&PdNum::from(*c))).collect()),
        PdObj::List(x) => PdObj::List(Rc::new(x.iter().map(|e| Rc::new(pd_deep_map(f, &*e))).collect())),
        _ => { panic!("wtf not deep"); }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PdKey {
    Num(PdTotalNum),
    String(Rc<Vec<char>>),
    List(Rc<Vec<Rc<PdKey>>>),
}

fn pd_key(obj: &PdObj) -> PdResult<PdKey> {
    match obj {
        PdObj::Num(x) => Ok(PdKey::Num(PdTotalNum(Rc::clone(x)))),
        PdObj::String(x) => Ok(PdKey::String(Rc::clone(x))),
        PdObj::List(x) => Ok(PdKey::List(Rc::new(x.iter().map(|k| Ok(Rc::new(pd_key(&**k)?))).collect::<PdResult<Vec<Rc<PdKey>>>>()?))),
        PdObj::Block(b) => Err(PdError::UnhashableBlock(b.code_repr())),
    }
}

/*
pub enum PdObj {
    Num(PdNum),
    String(Rc<Vec<char>>),
    List(Rc<Vec<Rc<PdObj>>>),
    Block(Rc<dyn Block>),
}
*/

// """Iterate a block, peeking at the stack top at the start and after each
// iteration, until a value repeats. Pop that value. Returns the list of (all
// distinct) elements peeked along the way and the final repeated value."""
fn pd_iterate(env: &mut Environment, func: &Rc<dyn Block>) -> PdResult<(Vec<Rc<PdObj>>, Rc<PdObj>)> {
    let mut acc: Vec<Rc<PdObj>> = Vec::new();
    let mut seen: HashSet<PdKey> = HashSet::new();

    loop {
        let obj = env.peek_result("iterate nothing to peek")?;
        let key = pd_key(&*obj)?;
        if seen.contains(&key) {
            env.pop_result("iterate final pop shouldn't fail lmao?")?;
            return Ok((acc, obj))
        }

        acc.push(obj);
        seen.insert(key);
        func.run(env)?;
    }
}

fn key_counter(a: &PdSeq) -> PdResult<HashMap<PdKey, usize>> {
    let mut ret = HashMap::new();
    for e in a.iter() {
        let key = pd_key(&*e)?;
        match ret.get_mut(&key) {
            Some(place) => { *place = *place + 1usize; }
            None => { ret.insert(key, 1usize); }
        }
    }
    Ok(ret)
}

fn key_set(a: &PdSeq) -> PdResult<HashSet<PdKey>> {
    let mut ret = HashSet::new();
    for e in a.iter() {
        ret.insert(pd_key(&*e)?);
    }
    Ok(ret)
}

fn pd_build_like(ty: PdSeqBuildType, x: Vec<Rc<PdObj>>) -> PdObj {
    if ty == PdSeqBuildType::NotString {
        return PdObj::List(Rc::new(x))
    }

    let mut chars: Vec<char> = Vec::new();
    let mut char_ok = true;
    for n in &x {
        match &**n {
            PdObj::Num(nn) => match &**nn {
                PdNum::Char(c) => { chars.push(c.to_u32().and_then(std::char::from_u32).expect("char, c'mon (pdbl)")); }
                _ => { char_ok = false; break }
            }
            _ => { char_ok = false; break }
        }
    }
    if char_ok {
        PdObj::from(chars)
    } else {
        PdObj::List(Rc::new(x))
    }
}

fn pd_seq_intersection(a: &PdSeq, b: &PdSeq) -> PdResult<PdObj> {
    let bty = a.build_type() & b.build_type();
    let mut counter = key_counter(b)?;
    let mut acc = Vec::new();
    for e in a.iter() {
        let key = pd_key(&e)?;
        match counter.get_mut(&key) {
            Some(place) => { if *place > 0 { acc.push(e); *place -= 1; } }
            None => {}
        }
    }
    Ok(pd_build_like(bty, acc))
}

fn pd_seq_union(a: &PdSeq, b: &PdSeq) -> PdResult<PdObj> {
    let bty = a.build_type() & b.build_type();
    let mut counter = key_counter(a)?;
    let mut acc = a.to_new_vec();
    for e in b.iter() {
        let key = pd_key(&e)?;
        match counter.get_mut(&key) {
            Some(place) => {
                if *place > 0 { *place -= 1; } else { acc.push(e); }
            }
            None => {
                acc.push(e);
            }
        }
    }
    Ok(pd_build_like(bty, acc))
}

fn pd_seq_set_difference(a: &PdSeq, b: &PdSeq) -> PdResult<PdObj> {
    let bty = a.build_type() & b.build_type();
    let ks = key_set(b)?;
    let mut acc = Vec::new();
    for e in a.iter() {
        if !ks.contains(&pd_key(&e)?) {
            acc.push(e);
        }
    }
    Ok(pd_build_like(bty, acc))
}

fn pd_seq_symmetric_difference(a: &PdSeq, b: &PdSeq) -> PdResult<PdObj> {
    let bty = a.build_type() & b.build_type();
    let mut acc = Vec::new();

    let bks = key_set(b)?;
    for e in a.iter() {
        if !bks.contains(&pd_key(&e)?) { acc.push(e); }
    }

    let aks = key_set(b)?;
    for e in b.iter() {
        if !aks.contains(&pd_key(&e)?) { acc.push(e); }
    }

    Ok(pd_build_like(bty, acc))
}

/*
def pd_seq_intersection(a: PdSeq, b: PdSeq) -> PdSeq:
    counter = collections.Counter(pykey(e) for e in pd_iterable(b))
    acc: List[PdObject] = []
    for element in pd_iterable(a):
        key = pykey(element)
        if counter[key] > 0:
            acc.append(element)
            counter[key] -= 1
    return pd_build_like(a, acc)
def pd_seq_union(a: PdSeq, b: PdSeq) -> PdSeq:
    acc: List[PdObject] = list(pd_iterable(a))
    counter = collections.Counter(pykey(e) for e in pd_iterable(a))
    for element in pd_iterable(b):
        key = pykey(element)
        if counter[key] > 0:
            counter[key] -= 1
        else:
            acc.append(element)
    return pd_build_like(a, acc)
def pd_seq_difference(a: PdSeq, b: PdSeq) -> PdSeq:
    set_b = set(pykey(e) for e in pd_iterable(b))
    acc: List[PdObject] = []
    for element in pd_iterable(a):
        if pykey(element) not in set_b:
            acc.append(element)
    return pd_build_like(a, acc)
def pd_seq_symmetric_difference(a: PdSeq, b: PdSeq) -> PdSeq:
    set_a = collections.Counter(pykey(e) for e in pd_iterable(a))
    set_b = collections.Counter(pykey(e) for e in pd_iterable(b))
    acc: List[PdObject] = []
    for element in pd_iterable(a):
        if pykey(element) not in set_b:
            acc.append(element)
    for element in pd_iterable(b):
        if pykey(element) not in set_a:
            acc.append(element)
    return pd_build_like(a, acc)
*/

fn bi_iverson(b: bool) -> BigInt { if b { BigInt::from(1) } else { BigInt::from(0) } }

pub fn initialize(env: &mut Environment) {
    let plus_case = nn_n![a, b, a + b];
    let minus_case = nn_n![a, b, a - b];
    let times_case = nn_n![a, b, a * b];
    // TODO: signs...
    let div_case = nn_n![a, b, a / b];
    let mod_case = nn_n![a, b, a % b];
    let intdiv_case = nn_n![a, b, a.div_floor(b)];

    let bitand_case = nn_n![a, b, a & b];
    let bitor_case  = nn_n![a, b, a | b];
    let bitxor_case = nn_n![a, b, a ^ b];

    let inc_case   = n_n![a, a.add_const( 1)];
    let dec_case   = n_n![a, a.add_const(-1)];
    let inc2_case  = n_n![a, a.add_const( 2)];
    let dec2_case  = n_n![a, a.add_const(-2)];

    let ceil_case   = n_n![a, a.ceil()];
    let floor_case  = n_n![a, a.floor()];
    let abs_case    = n_n![a, a.abs()];
    let neg_case    = n_n![a, -a];
    let signum_case = n_n![a, a.signum()];

    let eq_case = nn_n![a, b, PdNum::Int(bi_iverson(a == b))];
    let lt_case = nn_n![a, b, PdNum::Int(bi_iverson(a < b))];
    let gt_case = nn_n![a, b, PdNum::Int(bi_iverson(a > b))];
    let min_case = nn_n![a, b, PdNum::clone(a.min(b))];
    let max_case = nn_n![a, b, PdNum::clone(a.max(b))];

    let min_seq_case = unary_seq_case(|_, a: &PdSeq| {
        let mut best: Option<Rc<PdObj>> = None;
        // i don't think i can simplify this easily because of the inner return Err
        for obj in a.iter() {
            match best {
                None => { best = Some(obj); }
                Some(old) => {
                    best = Some(match obj.partial_cmp(&old) {
                        Some(Ordering::Less) => obj,
                        Some(Ordering::Equal) => old,
                        Some(Ordering::Greater) => old,
                        None => { return Err(PdError::BadComparison); }
                    })
                }
            }
        }
        Ok(vec![best.ok_or(PdError::BadList("Min of empty list"))?])
    });
    let max_seq_case = unary_seq_case(|_, a: &PdSeq| {
        let mut best: Option<Rc<PdObj>> = None;
        // i don't think i can simplify this easily because of the inner return Err
        for obj in a.iter() {
            match best {
                None => { best = Some(obj); }
                Some(old) => {
                    best = Some(match obj.partial_cmp(&old) {
                        Some(Ordering::Less) => old,
                        Some(Ordering::Equal) => obj,
                        Some(Ordering::Greater) => obj,
                        None => { return Err(PdError::BadComparison); }
                    })
                }
            }
        }
        Ok(vec![best.ok_or(PdError::BadList("Min of empty list"))?])
    });

    let uncons_case = unary_seq_range_case(|_, a| {
        let (x, xs) = a.split_first().ok_or(PdError::BadList("Uncons of empty list"))?;
        Ok(vec![xs, x])
    });
    let first_case = unary_seq_range_case(|_, a| { Ok(vec![a.first().ok_or(PdError::BadList("First of empty list"))?.to_rc_pd_obj() ]) });
    let rest_case = unary_seq_range_case(|_, a| { Ok(vec![a.split_first().ok_or(PdError::BadList("Rest of empty list"))?.1]) });

    let unsnoc_case = unary_seq_range_case(|_, a| {
        let (x, xs) = a.split_last().ok_or(PdError::BadList("Unsnoc of empty list"))?;
        Ok(vec![xs, x])
    });
    let last_case = unary_seq_range_case(|_, a| { Ok(vec![a.last().ok_or(PdError::BadList("Last of empty list"))?.to_rc_pd_obj() ]) });
    let butlast_case = unary_seq_range_case(|_, a| { Ok(vec![a.split_last().ok_or(PdError::BadList("Butlast of empty list"))?.1]) });

    let mut add_cases = |name: &str, cases: Vec<Rc<dyn Case>>| {
        env.variables.insert(name.to_string(), Rc::new(PdObj::Block(Rc::new(CasedBuiltIn {
            name: name.to_string(),
            cases,
        }))));
    };

    let map_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_block,
        coerce2: seq_range,
        func: |env, a, b| {
            Ok(vec![pd_list(pd_map(env, a, b.iter())?)])
        }
    });

    let square_case   : Rc<dyn Case> = Rc::new(UnaryNumCase { func: |_, a| Ok(vec![Rc::new(PdObj::from(a * a))]) });

    let space_join_case = unary_seq_range_case(|env, a| { Ok(vec![Rc::new(PdObj::from(a.iter().map(|x| env.to_string(&x)).collect::<Vec<String>>().join(" ")))]) });

    let index_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_seq,
        coerce2: num_to_isize,
        func: |_, seq, index| {
            Ok(vec![seq.pythonic_index(*index).ok_or(PdError::IndexError(index.to_string()))?.to_rc_pd_obj()])
        },
    });
    let len_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| { Ok(vec![Rc::new(PdObj::from(seq.len()))]) },
    });
    let down_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: seq_range,
        func: |_, seq| {
            Ok(vec![seq.rev_copy().to_rc_pd_obj()])
        },
    });
    let lt_slice_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_seq,
        coerce2: num_to_isize,
        func: |_, seq, index| {
            Ok(vec![seq.pythonic_split_left(*index).to_rc_pd_obj()])
        },
    });
    let mod_slice_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_seq,
        coerce2: num_to_isize,
        func: |_, seq, index| {
            Ok(vec![seq.pythonic_mod_slice(*index)?.to_rc_pd_obj()])
        },
    });
    let ge_slice_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_seq,
        coerce2: num_to_isize,
        func: |_, seq, index| {
            Ok(vec![seq.pythonic_split_right(*index).to_rc_pd_obj()])
        },
    });
    let seq_split_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_seq,
        coerce2: num_to_nn_usize,
        func: |_, seq, size| {
            Ok(vec![pd_list(slice_util::split_slice(seq.to_new_vec().as_slice(), *size, true).iter().map(|s| pd_list(s.to_vec())).collect())])
        },
    });
    let str_split_by_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_string,
        coerce2: just_string,
        func: |_, seq, tok| {
            Ok(vec![pd_list(slice_util::split_slice_by(seq.as_slice(), tok.as_slice()).iter().map(|s| Rc::new(PdObj::String(Rc::new(s.to_vec())))).collect())])
        },
    });
    let seq_split_by_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_seq,
        coerce2: just_seq,
        func: |_, seq, tok| {
            Ok(vec![pd_list(slice_util::split_slice_by(seq.to_new_vec().as_slice(), tok.to_new_vec().as_slice()).iter().map(|s| pd_list(s.to_vec())).collect())])
        },
    });

    let cat_list_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: list_singleton,
        coerce2: list_singleton,
        func: |_, seq1: &Rc<Vec<Rc<PdObj>>>, seq2: &Rc<Vec<Rc<PdObj>>>| {
            let mut v = Vec::new();
            v.extend((&**seq1).iter().cloned());
            v.extend((&**seq2).iter().cloned());

            Ok(vec![Rc::new(PdObj::List(Rc::new(v)))])
        },
    });

    let intersection_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: seq_range,
        coerce2: seq_range,
        func: |_, seq1, seq2| {
            Ok(vec![Rc::new(pd_seq_intersection(seq1, seq2)?)])
        },
    });
    let union_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: seq_range,
        coerce2: seq_range,
        func: |_, seq1, seq2| {
            Ok(vec![Rc::new(pd_seq_union(seq1, seq2)?)])
        },
    });
    let set_difference_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: seq_range,
        coerce2: seq_range,
        func: |_, seq1, seq2| {
            Ok(vec![Rc::new(pd_seq_set_difference(seq1, seq2)?)])
        },
    });
    let symmetric_difference_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: seq_range,
        coerce2: seq_range,
        func: |_, seq1, seq2| {
            Ok(vec![Rc::new(pd_seq_symmetric_difference(seq1, seq2)?)])
        },
    });

    add_cases("+", cc![plus_case, cat_list_case]);
    add_cases("-", cc![minus_case, set_difference_case]);
    add_cases("*", cc![times_case]);
    add_cases("/", cc![div_case, seq_split_case, str_split_by_case, seq_split_by_case]);
    add_cases("%", cc![mod_case, mod_slice_case, map_case]);
    add_cases("÷", cc![intdiv_case]);
    add_cases("&", cc![bitand_case, intersection_case]);
    add_cases("|", cc![bitor_case, union_case]);
    add_cases("^", cc![bitxor_case, symmetric_difference_case]);
    add_cases("(", cc![dec_case, uncons_case]);
    add_cases(")", cc![inc_case, unsnoc_case]);
    add_cases("=", cc![eq_case, index_case]);
    add_cases("<", cc![lt_case, lt_slice_case]);
    add_cases(">", cc![gt_case, ge_slice_case]);
    add_cases("<m", cc![min_case]);
    add_cases(">m", cc![max_case]);
    add_cases("Õ", cc![min_case]);
    add_cases("Ã", cc![max_case]);
    add_cases("<r", cc![min_seq_case]);
    add_cases(">r", cc![max_seq_case]);
    add_cases("D", cc![down_case]);
    add_cases("L", cc![abs_case, len_case]);
    add_cases("M", cc![neg_case]);
    add_cases("U", cc![signum_case]);
    add_cases("Œ", cc![min_seq_case]);
    add_cases("Æ", cc![max_seq_case]);
    add_cases("‹", cc![floor_case, first_case]);
    add_cases("›", cc![ceil_case, last_case]);
    add_cases("«", cc![dec2_case, butlast_case]);
    add_cases("»", cc![inc2_case, rest_case]);
    add_cases("²", cc![square_case]);
    add_cases(" r", cc![space_join_case]);

    add_cases(":",   vec![juggle!(a -> a, a)]);
    add_cases(":p",  vec![juggle!(a, b -> a, b, a, b)]);
    add_cases(":a",  vec![juggle!(a, b -> a, b, a)]);
    add_cases("\\",  vec![juggle!(a, b -> b, a)]);
    add_cases("\\a", vec![juggle!(a, b, c -> c, b, a)]);
    add_cases("\\i", vec![juggle!(a, b, c -> c, a, b)]);
    add_cases("\\o", vec![juggle!(a, b, c -> b, c, a)]);

    add_cases(";",   vec![juggle!(_a -> )]);
    add_cases(";o",  vec![juggle!(_a, b, c -> b, c)]);
    add_cases(";p",  vec![juggle!(_a, _b, c -> c)]);
    add_cases(";a",  vec![juggle!(_a, b, _c -> b)]);
    add_cases("¸",   vec![juggle!(_a, b -> b)]);

    let pack_one_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| Ok(vec![pd_list(vec![Rc::clone(a)])]) });
    add_cases("†", cc![pack_one_case]);
    let pack_two_case: Rc<dyn Case> = Rc::new(BinaryAnyCase { func: |_, a, b| Ok(vec![pd_list(vec![Rc::clone(a), Rc::clone(b)])]) });
    add_cases("‡", cc![pack_two_case]);
    let not_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| Ok(vec![Rc::new(PdObj::from(bi_iverson(!pd_truthy(a))))]) });
    add_cases("!", cc![not_case]);

    let sum_case   : Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| Ok(vec![Rc::new(PdObj::from(pd_deep_sum(a)?))]) });
    add_cases("Š", cc![sum_case]);

    let product_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| Ok(vec![Rc::new(PdObj::from(pd_deep_product(a)?))]) });
    add_cases("Þ", cc![product_case]);

    let average_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| Ok(vec![Rc::new(PdObj::from(pd_deep_sum(a)? / PdNum::Float(pd_deep_length(a)? as f64)))]) });
    add_cases("Av", cc![average_case]);

    let hypotenuse_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| Ok(vec![Rc::new(PdObj::from(pd_deep_square_sum(a)?.sqrt().ok_or(PdError::NumericError("sqrt in hypotenuse failed"))?))]) });
    add_cases("Hy", cc![hypotenuse_case]);

    let standard_deviation_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| Ok(vec![Rc::new(PdObj::from(pd_deep_standard_deviation(a)?))]) });
    add_cases("Sg", cc![standard_deviation_case]);

    let iterate_case: Rc<dyn Case> = Rc::new(UnaryCase { func: |env, block| Ok(vec![pd_list(pd_iterate(env, block)?.0)]), coerce: just_block });
    add_cases("I", cc![iterate_case]);

    // env.variables.insert("X".to_string(), Rc::new(PdObj::Int(3.to_bigint().unwrap())));
    env.short_insert("N", PdObj::from('\n'));
    env.short_insert("A", PdObj::from(10));
    env.short_insert("¹", PdObj::from(11));
    env.short_insert("∅", PdObj::from(0));
    env.short_insert("α", PdObj::from(1));
    env.short_insert("Ep", PdObj::from(1e-9));

    env.short_insert(" ", PdObj::Block(Rc::new(BuiltIn {
        name: "Nop".to_string(),
        func: |_env| Ok(()),
    })));
    env.short_insert("\n", PdObj::Block(Rc::new(BuiltIn {
        name: "Nop".to_string(),
        func: |_env| Ok(()),
    })));
    env.short_insert("[", PdObj::Block(Rc::new(BuiltIn {
        name: "Mark_stack".to_string(),
        func: |env| { env.mark_stack(); Ok(()) },
    })));
    env.short_insert("]", PdObj::Block(Rc::new(BuiltIn {
        name: "Pack".to_string(),
        func: |env| {
            let list = env.pop_until_stack_marker();
            env.push(pd_list(list));
            Ok(())
        },
    })));
    env.short_insert("¬", PdObj::Block(Rc::new(BuiltIn {
        name: "Pack_reverse".to_string(),
        func: |env| {
            let mut list = env.pop_until_stack_marker();
            list.reverse();
            env.push(pd_list(list));
            Ok(())
        },
    })));
    env.short_insert("~", PdObj::Block(Rc::new(BuiltIn {
        name: "Expand_or_eval".to_string(),
        func: |env| {
            match &*env.pop_result("~ failed")? {
                PdObj::Block(bb) => bb.run(env),
                PdObj::List(ls) => { env.extend_clone(ls); Ok(()) }
                _ => Err(PdError::BadArgument("~ can't handle".to_string())),
            }
        },
    })));
    env.short_insert("O", PdObj::Block(Rc::new(BuiltIn {
        name: "Print".to_string(),
        func: |env| {
            let obj = env.pop_result("O failed")?;
            print!("{}", env.to_string(&obj));
            Ok(())
        },
    })));
    env.short_insert("P", PdObj::Block(Rc::new(BuiltIn {
        name: "Print".to_string(),
        func: |env| {
            let obj = env.pop_result("P failed")?;
            println!("{}", env.to_string(&obj));
            Ok(())
        },
    })));
}

pub fn simple_eval(code: &str) -> Vec<Rc<PdObj>> {
    let mut env = Environment::new();
    initialize(&mut env);

    let block = CodeBlock::parse(code);

    match block.run(&mut env) {
        Ok(()) => {}
        Err(e) => { panic!("{:?}", e); }
    }

    env.stack
}
