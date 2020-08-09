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
mod string_util;
use crate::pdnum::{PdNum, PdTotalNum};
use crate::pderror::{PdError, PdResult, PdUnit};
use crate::input::{InputTrigger, ReadValue, EOFReader};
use crate::string_util::{str_class, int_groups, float_groups};

#[derive(Debug)]
pub struct TopEnvironment {
    x_stack: Vec<PdObj>,
    variables: HashMap<String, PdObj>,
    input_trigger: Option<InputTrigger>,
    reader: Option<EOFReader>,
}

#[derive(Debug)]
pub struct Environment {
    stack: Vec<PdObj>,
    marker_stack: Vec<usize>,
    shadow: Result<ShadowState, TopEnvironment>,
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
    fn pop(&mut self) -> Option<PdObj> {
        match self.shadow_type {
            ShadowType::Normal => {
                let res = self.env.pop();
                if res.is_some() { self.arity += 1; }
                res
            }
            ShadowType::Keep => {
                let res = self.env.stack.get(self.env.stack.len() - 1 - self.arity);
                if res.is_some() { self.arity += 1; }
                res.map(PdObj::clone)
            }
        }
    }
}

impl Environment {
    fn push(&mut self, obj: PdObj) {
        self.stack.push(obj)
    }
    // idk what the idiomatic way is yet
    // fn extend(&mut self, objs: Vec<PdObj>) {
    //     for obj in objs {
    //         self.push(obj)
    //     }
    // }
    fn extend(&mut self, objs: Vec<PdObj>) {
        for obj in objs {
            self.push(obj)
        }
    }
    fn extend_clone(&mut self, objs: &[PdObj]) {
        for obj in objs {
            self.push(PdObj::clone(obj))
        }
    }

    fn run_stack_trigger(&mut self) -> Option<PdObj> {
        match &mut self.shadow {
            Ok(inner) => inner.env.run_stack_trigger(),
            Err(top) => match (&mut top.reader, top.input_trigger) {
                (Some(r), Some(t)) => {
                    r.read(t).expect("io error in input trigger").map(|v| (PdObj::from(v)))
                }
                _ => None
            }
        }
    }

    fn pop(&mut self) -> Option<PdObj> {
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
                Ok(inner) => inner.pop(),
                Err(_) => self.run_stack_trigger()
            }
        }
    }
    fn pop_result(&mut self, err_msg: &str) -> PdResult<PdObj> {
        self.pop().ok_or(PdError::EmptyStack(err_msg.to_string()))
    }
    fn pop_n_result(&mut self, n: usize, err_msg: &str) -> Result<Vec<PdObj>, PdError> {
        let mut ret: Vec<PdObj> = Vec::new();
        for _ in 0..n {
            ret.push(self.pop_result(err_msg)?);
        }
        ret.reverse();
        Ok(ret)
    }

    fn peek_result(&mut self, err_msg: &str) -> PdResult<PdObj> {
        let ret = self.pop_result(err_msg)?;
        self.push(PdObj::clone(&ret));
        Ok(ret)
    }

    fn take_stack(&mut self) -> Vec<PdObj> {
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

    fn pop_until_stack_marker(&mut self) -> Vec<PdObj> {
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

    fn borrow_x_stack(&self) -> &Vec<PdObj> {
        match &self.shadow {
            Ok(inner) => inner.env.borrow_x_stack(),
            Err(top) => &top.x_stack,
        }
    }

    fn borrow_x_stack_mut(&mut self) -> &mut Vec<PdObj> {
        match &mut self.shadow {
            Ok(inner) => inner.env.borrow_x_stack_mut(),
            Err(top) => &mut top.x_stack,
        }
    }

    fn borrow_variables(&mut self) -> &mut HashMap<String, PdObj> {
        match &mut self.shadow {
            Ok(inner) => inner.env.borrow_variables(),
            Err(top) => &mut top.variables,
        }
    }

    fn push_x(&mut self, obj: PdObj) {
        self.borrow_x_stack_mut().push(obj)
    }

    fn push_yx(&mut self) {
        self.push_x(PdObj::from("INTERNAL Y FILLER -- YOU SHOULD NOT SEE THIS".to_string()));
        self.push_x(PdObj::from("INTERNAL X FILLER -- YOU SHOULD NOT SEE THIS".to_string()));
    }
    fn set_yx(&mut self, y: PdObj, x: PdObj) {
        let x_stack = self.borrow_x_stack_mut();
        let len = x_stack.len();
        x_stack[len - 2] = y;
        x_stack[len - 1] = x;
    }
    fn pop_yx(&mut self) {
        let x_stack = self.borrow_x_stack_mut();
        x_stack.pop().expect("m8 pop_yx");
        x_stack.pop().expect("m8 pop_yx");
    }

    fn short_insert(&mut self, name: &str, obj: impl Into<PdObj>) {
        self.borrow_variables().insert(name.to_string(), obj.into());
    }

    fn peek_x_stack(&self, depth: usize) -> Option<&PdObj> {
        let x_stack = self.borrow_x_stack();
        x_stack.get(x_stack.len().checked_sub(depth + 1)?)
    }

    fn get(&self, name: &str) -> Option<&PdObj> {
        match &self.shadow {
            Ok(inner) => inner.env.get(name),
            Err(top) => {
                match name {
                    "X" => self.peek_x_stack(0),
                    "Y" => self.peek_x_stack(1),
                    "Z" => self.peek_x_stack(2),
                    _ => top.variables.get(name)
                }
            }
        }
    }

    fn set_input_trigger(&mut self, t: Option<InputTrigger>) {
        match &mut self.shadow {
            Ok(inner) => inner.env.set_input_trigger(t),
            Err(top) => top.input_trigger = t,
        }
    }

    pub fn new_with_stdin() -> Environment {
        Environment {
            stack: Vec::new(),
            marker_stack: Vec::new(),
            shadow: Err(TopEnvironment {
                x_stack: Vec::new(),
                variables: HashMap::new(),
                input_trigger: None,
                reader: Some(EOFReader::new(Box::new(std::io::BufReader::new(std::io::stdin())))),
            }),
        }
    }

    pub fn new() -> Environment {
        Environment {
            stack: Vec::new(),
            marker_stack: Vec::new(),
            shadow: Err(TopEnvironment {
                x_stack: Vec::new(),
                variables: HashMap::new(),
                input_trigger: None,
                reader: None,
            }),
        }
    }

    fn to_string(&self, obj: &PdObj) -> String {
        match obj {
            PdObj::Num(n) => n.to_string(),
            PdObj::String(s) => s.iter().collect::<String>(),
            PdObj::List(v) => v.iter().map(|o| self.to_string(o)).collect::<Vec<String>>().join(""),
            PdObj::Block(b) => b.code_repr(),
        }
    }

    fn to_repr_string(&self, obj: &PdObj) -> String {
        match obj {
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
            marker_stack: vec![0],
            shadow: Ok(ShadowState { env: Box::new(env), arity: 0, shadow_type }),
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

fn sandbox(env: &mut Environment, func: &Rc<dyn Block>, args: Vec<PdObj>) -> Result<Vec<PdObj>, PdError> {
    env.run_on_bracketed_shadow(ShadowType::Normal, |inner| {
        inner.extend(args);
        func.run(inner)?;
        Ok(inner.take_stack())
    })
}
fn sandbox_truthy(env: &mut Environment, func: &Rc<dyn Block>, args: Vec<PdObj>) -> Result<bool, PdError> {
    let res = sandbox(env, func, args)?;
    let last = res.last().ok_or(PdError::EmptyStack("sandbox truthy bad".to_string()))?;
    Ok(pd_truthy(last))
}

#[derive(Debug, Clone)]
pub enum PdObj {
    Num(Rc<PdNum>),
    String(Rc<Vec<char>>),
    List(Rc<Vec<PdObj>>),
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

// 'static means it contains no references with a lifetime less than 'static
// all things that lack references satisfy that. weird
impl<T: Block + 'static> From<T> for PdObj {
    fn from(s: T) -> PdObj where T: 'static {
        PdObj::Block(Rc::new(s))
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
impl From<&str> for PdObj {
    fn from(s: &str) -> Self {
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
            ReadValue::List(v) => PdObj::List(Rc::new(v.into_iter().map(|x| (PdObj::from(x))).collect())),
        }
    }
}

fn bi_iverson(b: bool) -> BigInt { BigInt::from(if b { 1 } else { 0 }) }

impl PdObj {
    fn iverson(x: bool) -> Self {
        PdObj::from(bi_iverson(x))
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
    fn maybe_run_noncommutatively(&self, env: &mut Environment, args: &Vec<PdObj>) -> PdResult<Option<Vec<PdObj>>>;
}

struct UnaryAnyCase {
    func: fn(&mut Environment, &PdObj) -> PdResult<Vec<PdObj>>,
}
impl Case for UnaryAnyCase {
    fn arity(&self) -> usize { 1 }
    fn maybe_run_noncommutatively(&self, env: &mut Environment, args: &Vec<PdObj>) -> PdResult<Option<Vec<PdObj>>> {
        Ok(Some((self.func)(env, &args[0])?))
    }
}

struct BinaryAnyCase {
    func: fn(&mut Environment, &PdObj, &PdObj) -> PdResult<Vec<PdObj>>,
}
impl Case for BinaryAnyCase {
    fn arity(&self) -> usize { 2 }
    fn maybe_run_noncommutatively(&self, env: &mut Environment, args: &Vec<PdObj>) -> PdResult<Option<Vec<PdObj>>> {
        Ok(Some((self.func)(env, &args[0], &args[1])?))
    }
}

struct TernaryAnyCase {
    func: fn(&mut Environment, &PdObj, &PdObj, &PdObj) -> PdResult<Vec<PdObj>>,
}
impl Case for TernaryAnyCase {
    fn arity(&self) -> usize { 3 }
    fn maybe_run_noncommutatively(&self, env: &mut Environment, args: &Vec<PdObj>) -> PdResult<Option<Vec<PdObj>>> {
        Ok(Some((self.func)(env, &args[0], &args[1], &args[2])?))
    }
}

fn just_any(obj: &PdObj) -> Option<PdObj> {
    Some(PdObj::clone(obj))
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
fn num_to_clamped_usize(obj: &PdObj) -> Option<usize> {
    match obj {
        PdObj::Num(n) => Some(n.to_clamped_usize()),
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
    List(Rc<Vec<PdObj>>),
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

    fn to_rc_pd_obj(self) -> PdObj {
        match self {
            PdSeq::List(rc) => (PdObj::List(rc)),
            PdSeq::String(s) => (PdObj::String(s)),
            PdSeq::Range(_, _) => (PdObj::List(Rc::new(self.to_new_vec()))),
            // PdSeq::Range(a, b) => num_iter::range(BigInt::clone(a), BigInt::clone(b)).map(|x| (PdObj::from(x))).collect(),
        }
    }
}

pub enum PdSeqElement {
    ListElement(PdObj),
    Char(char),
    Int(BigInt),
}

impl PdSeqElement {
    fn to_rc_pd_obj(self) -> PdObj {
        match self {
            PdSeqElement::ListElement(rc) => rc,
            PdSeqElement::Char(c) => (PdObj::from(c)),
            PdSeqElement::Int(n) => (PdObj::from(n)),
        }
    }
}

pub enum PdIter<'a> {
    List(Iter<'a, PdObj>),
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
            PdSeq::List(v) => v.get(i).map(|e| PdSeqElement::ListElement(PdObj::clone(e))),
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
    fn to_new_vec(&self) -> Vec<PdObj> {
        match self {
            PdSeq::List(v) => (&**v).clone(),
            PdSeq::String(s) => s.iter().map(|x| (PdObj::from(*x))).collect(),
            PdSeq::Range(a, b) => num_iter::range(BigInt::clone(a), BigInt::clone(b)).map(|x| (PdObj::from(x))).collect(),
        }
    }

    fn first(&self) -> Option<PdSeqElement> {
        match self {
            PdSeq::List(v) => Some(PdSeqElement::ListElement(PdObj::clone(v.first()?))),
            PdSeq::String(s) => Some(PdSeqElement::Char(*s.first()?)),
            PdSeq::Range(a, b) => if a < b { Some(PdSeqElement::Int(BigInt::clone(a))) } else { None },
        }
    }

    fn split_first(&self) -> Option<(PdObj, PdObj)> {
        match self {
            PdSeq::List(v) => {
                let (x, xs) = v.split_first()?;
                Some((PdObj::clone(x), pd_list(xs.to_vec())))
            }
            PdSeq::String(s) => {
                let (x, xs) = s.split_first()?;
                Some(((PdObj::from(*x)), (PdObj::String(Rc::new(xs.to_vec())))))
            }
            PdSeq::Range(_, _) => {
                let v = self.to_new_vec();
                let (x, xs) = v.split_first()?;
                Some((PdObj::clone(x), pd_list(xs.to_vec())))
            }
        }
    }

    fn last(&self) -> Option<PdSeqElement> {
        match self {
            PdSeq::List(v) => Some(PdSeqElement::ListElement(PdObj::clone(v.last()?))),
            PdSeq::String(s) => Some(PdSeqElement::Char(*s.last()?)),
            PdSeq::Range(a, b) => if a < b { Some(PdSeqElement::Int(b - 1)) } else { None },
        }
    }

    fn split_last(&self) -> Option<(PdObj, PdObj)> {
        match self {
            PdSeq::List(v) => {
                let (x, xs) = v.split_last()?;
                Some((PdObj::clone(x), pd_list(xs.to_vec())))
            }
            PdSeq::String(s) => {
                let (x, xs) = s.split_last()?;
                Some((PdObj::from(*x), (PdObj::String(Rc::new(xs.to_vec())))))
            }
            PdSeq::Range(_, _) => {
                let v = self.to_new_vec();
                let (x, xs) = v.split_last()?;
                Some((PdObj::clone(x), pd_list(xs.to_vec())))
            }
        }
    }
}

impl Iterator for PdIter<'_> {
    type Item = PdObj;

    fn next(&mut self) -> Option<PdObj> {
        match self {
            PdIter::List(it) => it.next().map(PdObj::clone),
            PdIter::String(cs) => cs.next().map(|x| (PdObj::from(*x))),
            PdIter::Range(rs) => rs.next().map(|x| (PdObj::from(x))),
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
// TODO: hrmm do we want to make this take PdObj
// honestly we should just get rid of that extra Rc, it's pretty wasteful
fn list_singleton(obj: &PdObj) -> Option<Rc<Vec<PdObj>>> {
    match obj {
        PdObj::Num(n) => Some(Rc::new(vec![(PdObj::from(PdNum::clone(&**n)))])),
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
fn seq_num_singleton(obj: &PdObj) -> Option<PdSeq> {
    match obj {
        PdObj::Num(_) => Some(PdSeq::List(Rc::new(vec![PdObj::clone(obj)]))),
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
    func: fn(&mut Environment, &PdNum) -> PdResult<Vec<PdObj>>,
}
impl Case for UnaryNumCase {
    fn arity(&self) -> usize { 1 }
    fn maybe_run_noncommutatively(&self, env: &mut Environment, args: &Vec<PdObj>) -> PdResult<Option<Vec<PdObj>>> {
        match &args[0] {
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
    func: fn(&mut Environment, &T) -> PdResult<Vec<PdObj>>,
}
impl<T> Case for UnaryCase<T> {
    fn arity(&self) -> usize { 1 }
    fn maybe_run_noncommutatively(&self, env: &mut Environment, args: &Vec<PdObj>) -> PdResult<Option<Vec<PdObj>>> {
        match (self.coerce)(&args[0]) {
            Some(a) => {
                Ok(Some((self.func)(env, &a)?))
            }
            _ => Ok(None)
        }
    }
}
fn unary_num_case(func: fn(&mut Environment, &Rc<PdNum>) -> PdResult<Vec<PdObj>>) -> Rc<dyn Case> {
    Rc::new(UnaryCase { coerce: just_num, func })
}

struct BinaryCase<T1, T2> {
    coerce1: fn(&PdObj) -> Option<T1>,
    coerce2: fn(&PdObj) -> Option<T2>,
    func: fn(&mut Environment, &T1, &T2) -> PdResult<Vec<PdObj>>,
}
impl<T1, T2> Case for BinaryCase<T1, T2> {
    fn arity(&self) -> usize { 2 }
    fn maybe_run_noncommutatively(&self, env: &mut Environment, args: &Vec<PdObj>) -> PdResult<Option<Vec<PdObj>>> {
        match ((self.coerce1)(&args[0]), (self.coerce2)(&args[1])) {
            (Some(a), Some(b)) => {
                Ok(Some((self.func)(env, &a, &b)?))
            }
            _ => Ok(None)
        }
    }
}
fn binary_num_case(func: fn(&mut Environment, &Rc<PdNum>, &Rc<PdNum>) -> PdResult<Vec<PdObj>>) -> Rc<dyn Case> {
    Rc::new(BinaryCase { coerce1: just_num, coerce2: just_num, func })
}
fn unary_seq_case(func: fn(&mut Environment, &PdSeq) -> PdResult<Vec<PdObj>>) -> Rc<dyn Case> {
    Rc::new(UnaryCase { coerce: just_seq, func })
}
fn unary_seq_range_case(func: fn(&mut Environment, &PdSeq) -> PdResult<Vec<PdObj>>) -> Rc<dyn Case> {
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
        let mut accumulated_args: Vec<PdObj> = Vec::new();
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
        let res = pd_deep_map(&|x| (self.func)(x, &self.other), &env.pop_result("deep binary op no stack")?);
        env.push(res);
        Ok(())
    }
    fn code_repr(&self) -> String {
        self.other.to_string() + "_???_binary_op"
    }
}

struct DeepZipBlock {
    func: fn(&PdNum, &PdNum) -> PdNum,
    name: String,
}
impl Debug for DeepZipBlock {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(fmt, "DeepZipBLock {{ func: ???, name: {:?} }}", self.name)
    }
}
impl Block for DeepZipBlock {
    fn run(&self, env: &mut Environment) -> PdUnit {
        let b = env.pop_result("deep zip no stack")?;
        let a = env.pop_result("deep zip no stack")?;
        let res = pd_deep_zip(&self.func, &a, &b);
        env.push(res);
        Ok(())
    }
    fn code_repr(&self) -> String {
        String::clone(&self.name) + "_deep_zip"
    }
}

struct DeepCharToCharBlock {
    func: fn(char) -> char,
    name: String,
}
impl Debug for DeepCharToCharBlock {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(fmt, "DeepCharToCharBlock {{ func: ???, name: {:?} }}", self.name)
    }
}
impl Block for DeepCharToCharBlock {
    fn run(&self, env: &mut Environment) -> PdUnit {
        let a = env.pop_result("deep char to char no stack")?;
        let res = pd_deep_char_to_char(&self.func, &a);
        env.push(res);
        Ok(())
    }
    fn code_repr(&self) -> String {
        String::clone(&self.name) + "_deep_char_to_char"
    }
}

fn yx_loop<F>(env: &mut Environment, it: PdIter, mut body: F) -> PdUnit where F: FnMut(&mut Environment, usize, PdObj) -> PdUnit {
    env.push_yx();
    let mut ret = Ok(());
    for (i, obj) in it.enumerate() {
        env.set_yx(PdObj::from(i), PdObj::clone(&obj));
        match body(env, i, obj) {
            Ok(()) => {}
            Err(PdError::Break) => break,
            Err(PdError::Continue) => {}
            Err(e) => { ret = Err(e); break; }
        }
    }
    env.pop_yx();
    ret
}

fn pd_each(env: &mut Environment, func: &Rc<dyn Block>, it: PdIter) -> PdUnit {
    yx_loop(env, it, |env, _, obj| {
        env.push(obj);
        func.run(env)
    })
}

fn pd_xloop(env: &mut Environment, func: &Rc<dyn Block>, it: PdIter) -> PdUnit {
    yx_loop(env, it, |env, _, _| {
        func.run(env)
    })
}

fn pd_flatmap_foreach<F>(env: &mut Environment, func: &Rc<dyn Block>, mut body: F, it: PdIter) -> PdUnit where F: FnMut(PdObj) -> PdUnit {
    yx_loop(env, it, |env, _, obj| {
        for e in sandbox(env, &func, vec![obj])? {
            body(e)?
        }
        Ok(())
    })
}

// TODO: these should build-like, huh.
fn pd_map(env: &mut Environment, func: &Rc<dyn Block>, it: PdIter) -> PdResult<Vec<PdObj>> {
    let mut acc = Vec::new();
    pd_flatmap_foreach(env, func, |o| { acc.push(o); Ok(()) }, it)?;
    Ok(acc)
}

// TODO: wot, this isn't an xy loop in Paradoc proper?
fn pd_scan(env: &mut Environment, func: &Rc<dyn Block>, it: PdIter) -> PdResult<Vec<PdObj>> {
    let mut acc: Option<PdObj> = None;
    let mut ret: Vec<PdObj> = Vec::new();

    for e in it {
        let cur = match acc {
            None => e,
            Some(a) => {
                // pop consumes the sandbox result since we don't use it any more
                sandbox(env, func, vec![a, e])?.pop().ok_or(PdError::EmptyReduceIntermediate)?
            }
        };
        ret.push(PdObj::clone(&cur));
        acc = Some(cur);
    }

    Ok(ret)
}

// TODO: I don't think F should need to borrow &B, but it's tricky.
fn pd_flat_fold_with_short_circuit<B, F>(env: &mut Environment, func: &Rc<dyn Block>, init: B, body: F, it: PdIter) -> PdResult<B> where F: Fn(&B, PdObj) -> PdResult<(bool, B)> {
    let mut acc = init;
    pd_flatmap_foreach(env, func, |o| {
        let (do_break, acc2) = body(&acc, o)?;
        acc = acc2;
        if do_break { Err(PdError::Break) } else { Ok(()) }
    }, it)?;
    Ok(acc)
}

fn pd_flat_fold<B, F>(env: &mut Environment, func: &Rc<dyn Block>, init: B, body: F, it: PdIter) -> PdResult<B> where F: Fn(&B, PdObj) -> PdResult<B> {
    let mut acc = init;
    pd_flatmap_foreach(env, func, |o| { acc = body(&acc, o)?; Ok(()) }, it)?;
    Ok(acc)
}

enum FilterType { Filter, Reject }
impl FilterType {
    fn accept(&self, b: bool) -> bool {
        match self {
            FilterType::Filter => b,
            FilterType::Reject => !b,
        }
    }
}

fn pd_filter(env: &mut Environment, func: &Rc<dyn Block>, it: PdIter, fty: FilterType) -> PdResult<Vec<PdObj>> {
    let mut acc = Vec::new();
    yx_loop(env, it, |env, _, obj| {
        if fty.accept(sandbox_truthy(env, &func, vec![PdObj::clone(&obj)])?) {
            acc.push(obj)
        }
        Ok(())
    })?;
    Ok(acc)
}

fn pd_filter_indices(env: &mut Environment, func: &Rc<dyn Block>, it: PdIter, fty: FilterType) -> PdResult<Vec<PdObj>> {
    let mut acc = Vec::new();
    yx_loop(env, it, |env, i, obj| {
        if fty.accept(sandbox_truthy(env, &func, vec![obj])?) {
            acc.push(PdObj::from(i))
        }
        Ok(())
    })?;
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
    let opt_seq = seq_range(&env.pop_result(format!("{} no stack", name).as_str())?);
    opt_seq.ok_or(PdError::BadArgument(format!("{} coercion fail", name)))
}

#[derive(Debug)]
struct BindBlock {
    body: Rc<dyn Block>,
    bound_object: PdObj,
}
impl Block for BindBlock {
    fn run(&self, env: &mut Environment) -> PdUnit {
        env.push(PdObj::clone(&self.bound_object));
        self.body.run(env)
    }
    fn code_repr(&self) -> String {
        self.body.code_repr() + "_bind"
    }
}

#[derive(Debug)]
struct UnderBindBlock {
    body: Rc<dyn Block>,
    bound_object: PdObj,
}
impl Block for UnderBindBlock {
    fn run(&self, env: &mut Environment) -> PdUnit {
        let skip = env.pop_result("underbind skip fail")?;
        env.push(PdObj::clone(&self.bound_object));
        env.push(skip);
        self.body.run(env)
    }
    fn code_repr(&self) -> String {
        self.body.code_repr() + "_bind"
    }
}

#[derive(Debug, PartialEq)]
pub enum RcLeader {
    Lit(PdObj),
    Var(Rc<String>),
}

#[derive(Debug)]
pub struct CodeBlock(Vec<lex::Trailer>, Vec<RcToken>);

#[derive(Debug, PartialEq)]
pub struct RcToken(pub RcLeader, pub Vec<lex::Trailer>);

fn rcify(tokens: Vec<lex::Token>) -> Vec<RcToken> {
    // for .. in .., which is implicitly into_iter(), can move ownership out of the array
    // (iter() borrows elements only, but we are consuming tokens here)
    tokens.into_iter().map(|lex::Token(leader, mut trailer)| {
        let rcleader = match leader {
            lex::Leader::StringLit(s) => {
                RcLeader::Lit(PdObj::from(s))
            }
            lex::Leader::IntLit(n) => {
                RcLeader::Lit(PdObj::from(n))
            }
            lex::Leader::CharLit(c) => {
                RcLeader::Lit(PdObj::from(c))
            }
            lex::Leader::FloatLit(f) => {
                RcLeader::Lit(PdObj::from(f))
            }
            lex::Leader::Block(ty, t, b) => {
                match ty {
                    lex::BlockType::Normal => {}
                    lex::BlockType::Map => {
                        trailer.insert(0, lex::Trailer("map".to_string()))
                    }
                    lex::BlockType::Each => {
                        trailer.insert(0, lex::Trailer("each".to_string()))
                    }
                    lex::BlockType::Filter => {
                        trailer.insert(0, lex::Trailer("filter".to_string()))
                    }
                    lex::BlockType::Xloop => {
                        trailer.insert(0, lex::Trailer("xloop".to_string()))
                    }
                    lex::BlockType::Zip => {
                        trailer.insert(0, lex::Trailer("zip".to_string()))
                    }
                    lex::BlockType::Loop => {
                        trailer.insert(0, lex::Trailer("loop".to_string()))
                    }
                }
                RcLeader::Lit(PdObj::Block(Rc::new(CodeBlock(t, rcify(b)))))
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

fn obb(name: &'static str, bb: &Rc<dyn Block>, f: fn(&mut Environment, &Rc<dyn Block>) -> PdUnit) -> PdResult<(PdObj, bool)> {
    let body: Rc<dyn Block> = Rc::clone(bb);
    Ok(((PdObj::Block(Rc::new(OneBodyBlock { name, body, f }))), false))
}
fn apply_trailer(outer_env: &mut Environment, obj: &PdObj, trailer0: &lex::Trailer) -> PdResult<(PdObj, bool)> {
    let mut trailer: &str = trailer0.0.as_ref();
    trailer = trailer.strip_prefix('_').unwrap_or(trailer);

    match obj {
        PdObj::Num(n) => match trailer {
            "m" | "minus" => { Ok(((PdObj::from(-&**n)), false)) }
            "h" | "hundred" => { Ok(((PdObj::from(n.mul_const(100))), false)) }
            "k" | "thousand" => { Ok(((PdObj::from(n.mul_const(1000))), false)) }
            "p" | "power" => {
                let exponent: PdNum = PdNum::clone(n);
                Ok(((PdObj::Block(Rc::new(DeepBinaryOpBlock { func: |a, b| a.pow_num(b), other: exponent }))), false))
            }
            _ => Err(PdError::InapplicableTrailer(format!("{:?} on {:?}", trailer, obj)))
        }
        PdObj::Block(bb) => match trailer {
            "" => Ok((PdObj::clone(obj), true)),
            "b" | "bind" => {
                let b = outer_env.pop_result("bind nothing to bind")?;
                Ok(((PdObj::Block(Rc::new(BindBlock {
                    body: Rc::clone(bb),
                    bound_object: b,
                }))), true))
            }
            "e" | "each" => obb("each", bb, |env, body| {
                let seq = pop_seq_range_for(env, "each")?;
                pd_each(env, body, seq.iter())
            }),
            "x" | "xloop" => obb("xloop", bb, |env, body| {
                let seq = pop_seq_range_for(env, "xloop")?;
                pd_xloop(env, body, seq.iter())
            }),
            "z" | "zip" => obb("zip", bb, |_env, _body| {
                panic!("zip not implemented");
            }),
            "l" | "loop" => obb("loop", bb, |_env, _body| {
                panic!("loop not implemented");
            }),
            "m" | "map" => obb("map", bb, |env, body| {
                let seq = pop_seq_range_for(env, "map")?;
                let res = pd_map(env, body, seq.iter())?;
                env.push(pd_list(res));
                Ok(())
            }),
            "f" | "filter" => obb("filter", bb, |env, body| {
                let seq = pop_seq_range_for(env, "filter")?;
                let res = pd_filter(env, body, seq.iter(), FilterType::Filter)?;
                env.push(pd_list(res));
                Ok(())
            }),
            "j" | "reject" => obb("reject", bb, |env, body| {
                let seq = pop_seq_range_for(env, "reject")?;
                let res = pd_filter(env, body, seq.iter(), FilterType::Reject)?;
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
                env.push(PdObj::from(res));
                Ok(())
            }),
            "þ" | "product" => obb("product", bb, |env, body| {
                let seq = pop_seq_range_for(env, "product")?;
                let res = pd_flat_fold(env, body, PdNum::from(0),
                    |acc, o| { Ok(acc * &pd_deep_product(&o)?) }, seq.iter())?;
                env.push(PdObj::from(res));
                Ok(())
            }),
            "â" | "all" => obb("all", bb, |env, body| {
                let seq = pop_seq_range_for(env, "all")?;
                let res = pd_flat_fold_with_short_circuit(env, body, true,
                    |_, o| { if pd_truthy(&o) { Ok((true, false)) } else { Ok((false, true)) } }, seq.iter())?;
                env.push(PdObj::iverson(res));
                Ok(())
            }),
            "v" | "bindmap" | "vectorize" => obb("bindmap", bb, |env, body| {
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
            "y" | "mapbind" => obb("mapbind", bb, |env, body| {
                let seq = pop_seq_range_for(env, "mapbind")?;

                let b = env.pop_result("mapbind nothing to bind")?;
                let bb: Rc<dyn Block> = Rc::new(UnderBindBlock {
                    body: Rc::clone(body),
                    bound_object: b,
                });

                let res = pd_map(env, &bb, seq.iter())?;
                env.push(pd_list(res));
                Ok(())
            }),
            "s" | "scan" => obb("scan", bb, |env, body| {
                let seq = pop_seq_range_for(env, "scan")?;
                let res = pd_scan(env, body, seq.iter())?;
                env.push(pd_list(res));
                Ok(())
            }),
            "w" | "deepmap" => obb("deepmap", bb, |env, body| {
                let seq = pop_seq_range_for(env, "map")?.to_rc_pd_obj();
                let res = pd_deep_map_block(env, body, seq)?;
                env.extend(res);
                Ok(())
            }),
            "ø" | "organize" => obb("organize", bb, |env, body| {
                let seq = pop_seq_range_for(env, "map")?;
                let res = pd_organize_by(&seq, pd_key_projector(env, body))?;
                env.push(res);
                Ok(())
            }),

            _ => Err(PdError::InapplicableTrailer(format!("{:?} on {:?}", trailer, obj)))
        }
        _ => Err(PdError::InapplicableTrailer(format!("{:?} on {:?}", trailer, obj)))
    }
}

fn apply_all_trailers(env: &mut Environment, mut obj: PdObj, mut reluctant: bool, trailer: &[lex::Trailer]) -> PdResult<(PdObj, bool)> {
    for t in trailer {
        let np = apply_trailer(env, &obj, t)?; // unwraps or returns Err from entire function
        obj = np.0;
        reluctant = np.1;
    }
    Ok((obj, reluctant))
}

fn lookup_and_break_trailers<'a, 'b>(env: &'a Environment, leader: &str, trailers: &'b[lex::Trailer]) -> Option<(&'a PdObj, &'b[lex::Trailer])> {

    let mut var: String = leader.to_string();

    let mut last_found: Option<(&'a PdObj, &'b[lex::Trailer])> = None;

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
                "i" | "input"       => { env.set_input_trigger(Some(InputTrigger::All       )); }
                "l" | "line"        => { env.set_input_trigger(Some(InputTrigger::Line      )); }
                "w" | "word"        => { env.set_input_trigger(Some(InputTrigger::Word      )); }
                "v" | "values"      => { env.set_input_trigger(Some(InputTrigger::Value     )); }
                "r" | "record"      => { env.set_input_trigger(Some(InputTrigger::Record    )); }
                "a" | "linearray"   => { env.set_input_trigger(Some(InputTrigger::AllLines  )); }
                "y" | "valuearray"  => { env.set_input_trigger(Some(InputTrigger::AllValues )); }
                "q" | "recordarray" => { env.set_input_trigger(Some(InputTrigger::AllRecords)); }
                _ => { panic!("unsupported init trailer"); }
            }
        }

        for RcToken(leader, trailer) in &self.1 {
            // println!("{:?} {:?}", leader, trailer);
            let (obj, reluctant) = match leader {
                RcLeader::Lit(obj) => {
                    apply_all_trailers(env, PdObj::clone(obj), true, trailer)?
                }
                RcLeader::Var(s) => {
                    let (obj, rest) = lookup_and_break_trailers(env, s, trailer).ok_or(PdError::UndefinedVariable(String::clone(s)))?;
                    let cobj = PdObj::clone(obj); // borrow checker to drop obj which unborrows env
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

fn apply_on(env: &mut Environment, obj: PdObj) -> PdUnit {
    match obj {
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
            Ok(vec![PdObj::from($x)])
        })
    };
}
macro_rules! nn_n {
    ($a:ident, $b:ident, $x:expr) => {
        binary_num_case(|_, a, b| {
            let $a: &PdNum = &**a;
            let $b: &PdNum = &**b;
            Ok(vec![PdObj::from($x)])
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

macro_rules! poc {
    ($($case:expr),*) => {
        vec![$( PdObj::clone(&$case) ),*];
    }
}

macro_rules! juggle {
    ($a:ident -> $($res:expr),*) => {
        { let v: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, $a| Ok(poc![ $( $res ),* ]) }); v }
    };
    ($a:ident, $b:ident -> $($res:expr),*) => {
        { let v: Rc<dyn Case> = Rc::new(BinaryAnyCase { func: |_, $a, $b| Ok(poc![ $( $res ),* ]) }); v }
    };
    ($a:ident, $b:ident, $c:ident -> $($res:expr),*) => {
        { let v: Rc<dyn Case> = Rc::new(TernaryAnyCase { func: |_, $a, $b, $c| Ok(poc![ $( $res ),* ]) }); v }
    };
}

fn pd_list(xs: Vec<PdObj>) -> PdObj { PdObj::List(Rc::new(xs)) }

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
        PdObj::List(Rc::new(x.iter().map(|e| PdObj::Num(Rc::new(e.clone()))).collect()))
    }
}

// note to self: you do need to borrow f because of recursion
fn pd_deep_map<F>(f: &F, x: &PdObj) -> PdObj
    where F: Fn(&PdNum) -> PdNum {

    match x {
        PdObj::Num(n) => PdObj::Num(Rc::new(f(n))),
        PdObj::String(s) => pd_build_if_string(s.iter().map(|c| f(&PdNum::from(*c))).collect()),
        PdObj::List(x) => PdObj::List(Rc::new(x.iter().map(|e| pd_deep_map(f, e)).collect())),
        _ => { panic!("wtf not deep"); }
    }
}

// TODO fix; this is terrible
// NOTE the caller should expand this cuz why not
fn pd_deep_map_block(env: &mut Environment, func: &Rc<dyn Block>, x: PdObj) -> PdResult<Vec<PdObj>> {

    match x {
        PdObj::Num(_) => sandbox(env, func, vec![x]),
        // TODO also fix
        PdObj::String(s) => {
            let mut acc: Vec<PdObj> = Vec::new();

            for e in s.iter() {
                acc.extend(pd_deep_map_block(env, func, PdObj::from(*e))?);
            }

            Ok(vec![pd_build_like(PdSeqBuildType::String, acc)])
        }
        PdObj::List(x) => {
            let mut acc: Vec<PdObj> = Vec::new();

            for e in x.iter() {
                acc.extend(pd_deep_map_block(env, func, PdObj::clone(e))?);
            }

            Ok(vec![PdObj::List(Rc::new(acc))])
        }
        _ => { panic!("wtf not deep"); }
    }
}


fn pd_deep_zip<F>(f: &F, a: &PdObj, b: &PdObj) -> PdObj
    where F: Fn(&PdNum, &PdNum) -> PdNum {

    if let (PdObj::Num(na), PdObj::Num(nb)) = (a, b) {
        PdObj::Num(Rc::new(f(na, nb)))
    } else if let (Some(sa), Some(sb)) = (seq_num_singleton(a), seq_num_singleton(b)) {
        pd_build_like(
            sa.build_type() & sb.build_type(),
            sa.iter().zip(sb.iter()).map(|(ea, eb)| pd_deep_zip(f, &ea, &eb)).collect()
        )
    } else {
        panic!("wtf not deep");
    }
}

fn pd_deep_char_to_char<F>(f: &F, a: &PdObj) -> PdObj
    where F: Fn(char) -> char {

    pd_deep_map(&|num: &PdNum| {
        num.to_char().map_or(PdNum::clone(num), |ch| PdNum::from(f(ch)))
    }, a)
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub enum PdKey {
    Num(PdTotalNum),
    String(Rc<Vec<char>>),
    List(Rc<Vec<Rc<PdKey>>>),
}

fn pd_key(obj: &PdObj) -> PdResult<PdKey> {
    match obj {
        PdObj::Num(x) => Ok(PdKey::Num(PdTotalNum(Rc::clone(x)))),
        PdObj::String(x) => Ok(PdKey::String(Rc::clone(x))),
        PdObj::List(x) => Ok(PdKey::List(Rc::new(x.iter().map(|k| Ok(Rc::new(pd_key(&*k)?))).collect::<PdResult<Vec<Rc<PdKey>>>>()?))),
        PdObj::Block(b) => Err(PdError::UnhashableBlock(b.code_repr())),
    }
}

/*
pub enum PdObj {
    Num(PdNum),
    String(Rc<Vec<char>>),
    List(Rc<Vec<PdObj>>),
    Block(Rc<dyn Block>),
}
*/

// """Iterate a block, peeking at the stack top at the start and after each
// iteration, until a value repeats. Pop that value. Returns the list of (all
// distinct) elements peeked along the way and the final repeated value."""
fn pd_iterate(env: &mut Environment, func: &Rc<dyn Block>) -> PdResult<(Vec<PdObj>, PdObj)> {
    let mut acc: Vec<PdObj> = Vec::new();
    let mut seen: HashSet<PdKey> = HashSet::new();

    loop {
        let obj = env.peek_result("iterate nothing to peek")?;
        let key = pd_key(&obj)?;
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
        let key = pd_key(&e)?;
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
        ret.insert(pd_key(&e)?);
    }
    Ok(ret)
}

fn pd_build_like(ty: PdSeqBuildType, x: Vec<PdObj>) -> PdObj {
    if ty == PdSeqBuildType::NotString {
        return PdObj::List(Rc::new(x))
    }

    let mut chars: Vec<char> = Vec::new();
    let mut char_ok = true;
    for n in &x {
        match &*n {
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

fn pd_key_projector<'a>(env: &'a mut Environment, func: &'a Rc<dyn Block>) -> impl FnMut(&PdObj) -> PdResult<PdKey> + 'a {
    move |x| -> PdResult<PdKey> {
        Ok(PdKey::List(Rc::new(
            sandbox(env, func, vec![PdObj::clone(x)])?
            .iter()
            .map(|e| -> PdResult<Rc<PdKey>> { Ok(Rc::new(pd_key(e)?)) })
            .collect::<PdResult<Vec<Rc<PdKey>>>>()?
        )))
    }
}

fn pd_organize_by<F>(seq: &PdSeq, mut proj: F) -> PdResult<PdObj> where F: FnMut(&PdObj) -> PdResult<PdKey> {
    let bty = seq.build_type();
    let mut key_order: Vec<PdKey> = Vec::new();
    let mut groups: HashMap<PdKey, Vec<PdObj>> = HashMap::new();

    for e in seq.iter() {
        let key = proj(&e)?;
        match groups.get_mut(&key) {
            Some(place) => { place.push(e); }
            None => {
                key_order.push(PdKey::clone(&key));
                groups.insert(key, vec![e]);
            }
        }
    }

    // groups.remove() gives us ownership
    Ok(pd_list(key_order.iter().map(|key| pd_build_like(bty, groups.remove(key).expect("organize internal fail"))).collect()))
}

fn pd_sort_by<F>(seq: &PdSeq, mut proj: F) -> PdResult<PdObj> where F: FnMut(&PdObj) -> PdResult<PdKey> {
    let bty = seq.build_type();
    // sort_by_cached_key is not enough because we also want to force the PdResult
    let mut keyed_vec = seq.iter().map(|e| Ok((proj(&e)?, e))).collect::<PdResult<Vec<(PdKey, PdObj)>>>()?;
    // and sort_by_key's key function wants us to give ownership of the key :-/
    keyed_vec.sort_by(|x, y| x.0.cmp(&y.0));

    Ok(pd_build_like(bty, keyed_vec.into_iter().map(|x| x.1).collect()))
}

// TODO if this stabilizes https://github.com/rust-lang/rust/issues/53485
fn pd_is_sorted_by<F>(seq: &PdSeq, mut proj: F, accept: fn(Ordering) -> bool) -> PdResult<bool> where F: FnMut(&PdObj) -> PdResult<PdKey> {
    let mut prev: Option<PdKey> = None;
    for e in seq.iter() {
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
    let trunc_case  = n_n![a, a.trunc()];

    let eq_case = nn_n![a, b, PdNum::Int(bi_iverson(a == b))];
    let lt_case = nn_n![a, b, PdNum::Int(bi_iverson(a < b))];
    let gt_case = nn_n![a, b, PdNum::Int(bi_iverson(a > b))];
    let min_case = nn_n![a, b, PdNum::clone(a.min(b))];
    let max_case = nn_n![a, b, PdNum::clone(a.max(b))];

    let min_seq_case = unary_seq_case(|_, a: &PdSeq| {
        let mut best: Option<PdObj> = None;
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
        let mut best: Option<PdObj> = None;
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
        env.short_insert(name, CasedBuiltIn {
            name: name.to_string(),
            cases,
        });
    };

    let map_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_block,
        coerce2: seq_range,
        func: |env, a, b| {
            Ok(vec![pd_list(pd_map(env, a, b.iter())?)])
        }
    });

    let filter_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_block,
        coerce2: seq_range,
        func: |env, a, b| {
            Ok(vec![pd_list(pd_filter(env, a, b.iter(), FilterType::Filter)?)])
        }
    });
    let filter_indices_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_block,
        coerce2: seq_range,
        func: |env, a, b| {
            Ok(vec![pd_list(pd_filter_indices(env, a, b.iter(), FilterType::Filter)?)])
        }
    });
    let reject_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_block,
        coerce2: seq_range,
        func: |env, a, b| {
            Ok(vec![pd_list(pd_filter(env, a, b.iter(), FilterType::Reject)?)])
        }
    });
    let reject_indices_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_block,
        coerce2: seq_range,
        func: |env, a, b| {
            Ok(vec![pd_list(pd_filter_indices(env, a, b.iter(), FilterType::Reject)?)])
        }
    });

    let square_case   : Rc<dyn Case> = Rc::new(UnaryNumCase { func: |_, a| Ok(vec![(PdObj::from(a * a))]) });

    let space_join_case = unary_seq_range_case(|env, a| { Ok(vec![(PdObj::from(a.iter().map(|x| env.to_string(&x)).collect::<Vec<String>>().join(" ")))]) });

    let index_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_seq,
        coerce2: num_to_isize,
        func: |_, seq, index| {
            Ok(vec![seq.pythonic_index(*index).ok_or(PdError::IndexError(index.to_string()))?.to_rc_pd_obj()])
        },
    });
    let len_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| { Ok(vec![(PdObj::from(seq.len()))]) },
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
            Ok(vec![pd_list(slice_util::split_slice_by(seq.as_slice(), tok.as_slice()).iter().map(|s| (PdObj::String(Rc::new(s.to_vec())))).collect())])
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
        func: |_, seq1: &Rc<Vec<PdObj>>, seq2: &Rc<Vec<PdObj>>| {
            let mut v = Vec::new();
            v.extend((&**seq1).iter().cloned());
            v.extend((&**seq2).iter().cloned());

            Ok(vec![(PdObj::List(Rc::new(v)))])
        },
    });

    let intersection_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: seq_range,
        coerce2: seq_range,
        func: |_, seq1, seq2| {
            Ok(vec![pd_seq_intersection(seq1, seq2)?])
        },
    });
    let union_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: seq_range,
        coerce2: seq_range,
        func: |_, seq1, seq2| {
            Ok(vec![pd_seq_union(seq1, seq2)?])
        },
    });
    let set_difference_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: seq_range,
        coerce2: seq_range,
        func: |_, seq1, seq2| {
            Ok(vec![pd_seq_set_difference(seq1, seq2)?])
        },
    });
    let symmetric_difference_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: seq_range,
        coerce2: seq_range,
        func: |_, seq1, seq2| {
            Ok(vec![pd_seq_symmetric_difference(seq1, seq2)?])
        },
    });

    let range_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_num,
        func: |_, num| {
            let n = num.to_bigint().ok_or(PdError::BadFloat)?;
            let vs = num_iter::range(BigInt::from(0), n).map(|x| PdObj::from(num.construct_like_self(x))).collect();
            Ok(vec![(PdObj::List(Rc::new(vs)))])
        },
    });
    let one_range_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_num,
        func: |_, num| {
            let n = num.to_bigint().ok_or(PdError::BadFloat)?;
            let vs = num_iter::range_inclusive(BigInt::from(1), n).map(|x| PdObj::from(num.construct_like_self(x))).collect();
            Ok(vec![pd_list(vs)])
        },
    });
    let zip_range_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![pd_list(seq.iter().enumerate().map(|(i, x)| pd_list(vec![PdObj::from(i), x])).collect())])
        },
    });
    let zip_one_range_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![pd_list(seq.iter().enumerate().map(|(i, x)| pd_list(vec![PdObj::from(i + 1), x])).collect())])
        },
    });

    let til_range_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_num,
        coerce2: just_num,
        func: |_, num1, num2| {
            let n1 = num1.to_bigint().ok_or(PdError::BadFloat)?;
            let n2 = num2.to_bigint().ok_or(PdError::BadFloat)?;
            let vs = num_iter::range(n1, n2).map(|x| PdObj::from(num1.construct_like_self(x))).collect();
            Ok(vec![pd_list(vs)])
        },
    });
    let to_range_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_num,
        coerce2: just_num,
        func: |_, num1, num2| {
            let n1 = num1.to_bigint().ok_or(PdError::BadFloat)?;
            let n2 = num2.to_bigint().ok_or(PdError::BadFloat)?;
            let vs = num_iter::range_inclusive(n1, n2).map(|x| PdObj::from(num1.construct_like_self(x))).collect();
            Ok(vec![pd_list(vs)])
        },
    });

    // FIXME
    let organize_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![pd_organize_by(seq, pd_key)?])
        },
    });
    let organize_by_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: seq_range,
        coerce2: just_block,
        func: |env, seq, block| {
            Ok(vec![pd_organize_by(seq, pd_key_projector(env, block))?])
        },
    });

    let sort_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![pd_sort_by(seq, pd_key)?])
        },
    });
    let sort_by_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: seq_range,
        coerce2: just_block,
        func: |env, seq, block| {
            Ok(vec![pd_sort_by(seq, pd_key_projector(env, block))?])
        },
    });

    let is_sorted_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![PdObj::iverson(pd_is_sorted_by(seq, pd_key, |x| x != Ordering::Greater)?)])
        },
    });
    let is_sorted_by_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: seq_range,
        coerce2: just_block,
        func: |env, seq, block| {
            Ok(vec![PdObj::iverson(pd_is_sorted_by(seq, pd_key_projector(env, block), |x| x != Ordering::Greater)?)])
        },
    });

    let is_strictly_increasing_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![PdObj::iverson(pd_is_sorted_by(seq, pd_key, |x| x == Ordering::Less)?)])
        },
    });
    let is_strictly_increasing_by_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: seq_range,
        coerce2: just_block,
        func: |env, seq, block| {
            Ok(vec![PdObj::iverson(pd_is_sorted_by(seq, pd_key_projector(env, block), |x| x == Ordering::Less)?)])
        },
    });
    let is_strictly_decreasing_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![PdObj::iverson(pd_is_sorted_by(seq, pd_key, |x| x == Ordering::Greater)?)])
        },
    });
    let is_strictly_decreasing_by_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: seq_range,
        coerce2: just_block,
        func: |env, seq, block| {
            Ok(vec![PdObj::iverson(pd_is_sorted_by(seq, pd_key_projector(env, block), |x| x == Ordering::Greater)?)])
        },
    });

    add_cases("+", cc![plus_case, cat_list_case, filter_case]);
    add_cases("-", cc![minus_case, set_difference_case, reject_case]);
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
    add_cases(",", cc![range_case, zip_range_case, filter_indices_case]);
    add_cases("J", cc![one_range_case, zip_one_range_case, reject_indices_case]);
    add_cases("…", cc![to_range_case]);
    add_cases("¨", cc![til_range_case]);
    add_cases("To", cc![to_range_case]);
    add_cases("Tl", cc![til_range_case]);

    add_cases("Ø", cc![organize_case, organize_by_case]);
    add_cases("$", cc![sort_case, sort_by_case]);
    add_cases("$p", cc![is_sorted_case, is_sorted_by_case]);
    add_cases("<p", cc![is_strictly_increasing_case, is_strictly_increasing_by_case]);
    add_cases(">p", cc![is_strictly_decreasing_case, is_strictly_decreasing_by_case]);

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

    let pack_one_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| Ok(vec![pd_list(vec![PdObj::clone(a)])]) });
    add_cases("†", cc![pack_one_case]);
    let pack_two_case: Rc<dyn Case> = Rc::new(BinaryAnyCase { func: |_, a, b| Ok(vec![pd_list(vec![PdObj::clone(a), PdObj::clone(b)])]) });
    add_cases("‡", cc![pack_two_case]);
    let not_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| Ok(vec![(PdObj::iverson(!pd_truthy(a)))]) });
    add_cases("!", cc![not_case]);

    let sum_case   : Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| Ok(vec![(PdObj::from(pd_deep_sum(a)?))]) });
    add_cases("Š", cc![sum_case]);

    let product_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| Ok(vec![(PdObj::from(pd_deep_product(a)?))]) });
    add_cases("Þ", cc![product_case]);

    let average_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| Ok(vec![(PdObj::from(pd_deep_sum(a)? / PdNum::Float(pd_deep_length(a)? as f64)))]) });
    add_cases("Av", cc![average_case]);

    let hypotenuse_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| Ok(vec![(PdObj::from(pd_deep_square_sum(a)?.sqrt().ok_or(PdError::NumericError("sqrt in hypotenuse failed"))?))]) });
    add_cases("Hy", cc![hypotenuse_case]);

    let standard_deviation_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| Ok(vec![(PdObj::from(pd_deep_standard_deviation(a)?))]) });
    add_cases("Sg", cc![standard_deviation_case]);

    // FIXME
    let string_to_int_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_string,
        func: |_, s: &Rc<Vec<char>>| Ok(vec![PdObj::from(s.iter().collect::<String>().parse::<BigInt>().map_err(|_| PdError::BadParse)?)]),
    });
    let iterate_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_block,
        func: |env, block| Ok(vec![pd_list(pd_iterate(env, block)?.0)]),
    });
    add_cases("I", cc![trunc_case, string_to_int_case, iterate_case]);

    let int_groups_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_string,
        func: |_, s: &Rc<Vec<char>>| Ok(vec![pd_list(int_groups(&s.iter().collect::<String>()).map(|i| PdObj::from(i)).collect())]),
    });
    add_cases("Ig", cc![int_groups_case]);

    let float_groups_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_string,
        func: |_, s: &Rc<Vec<char>>| Ok(vec![pd_list(float_groups(&s.iter().collect::<String>()).map(|i| PdObj::from(i)).collect())]),
    });
    add_cases("Fg", cc![float_groups_case]);

    let to_string_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |env, a| Ok(vec![PdObj::from(env.to_string(a))]) });
    add_cases("S", cc![to_string_case]);

    let replicate_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_any,
        coerce2: num_to_clamped_usize,
        func: |_, a, n| {
            // resize() or vec![_; ] just clones but i don't want that to be silent
            // idk if this is being paranoid, but we've come this far
            let mut vec = Vec::new();
            vec.resize_with(*n, || PdObj::clone(a));
            Ok(vec![pd_list(vec)])
        },
    });
    add_cases("°", cc![replicate_case]);


    // env.variables.insert("X".to_string(), (PdObj::Int(3.to_bigint().unwrap())));
    env.short_insert("N", '\n');
    env.short_insert("A", 10);
    env.short_insert("¹", 11);
    env.short_insert("∅", 0);
    env.short_insert("α", 1);
    env.short_insert("Ep", 1e-9);
    env.short_insert("Ua", str_class("A-Z"));
    env.short_insert("La", str_class("a-z"));

    env.short_insert("Da", str_class("0-9"));
    env.short_insert("Ua", str_class("A-Z"));
    env.short_insert("La", str_class("a-z"));
    env.short_insert("Aa", str_class("A-Za-z"));

    // # Non-breaking space (U+00A0)
    env.short_insert("\u{a0}", ' ');
    env.short_insert("␣", ' ');

    env.short_insert("Å", str_class("A-Z"));
    env.short_insert("Åa", str_class("a-zA-Z"));
    // env.short_insert("Åb", case_double("BCDFGHJKLMNPQRSTVWXZ"));
    // env.short_insert("Åc", case_double("BCDFGHJKLMNPQRSTVWXYZ"));
    env.short_insert("Åd", str_class("9-0"));
    env.short_insert("Åf", str_class("A-Za-z0-9+/"));
    env.short_insert("Åh", str_class("0-9A-F"));
    env.short_insert("Åi", str_class("A-Za-z0-9_"));
    env.short_insert("Åj", str_class("a-zA-Z0-9_"));
    env.short_insert("Ål", str_class("z-a"));
    env.short_insert("Åm", "()<>[]{}");
    env.short_insert("Åp", str_class(" -~"));
    // env.short_insert("Åq", case_double("QWERTYUIOP"));
    // env.short_insert("Ås", case_double("ASDFGHJKL"));
    env.short_insert("Åt", str_class("0-9A-Z"));
    env.short_insert("Åu", str_class("Z-A"));
    // env.short_insert("Åv", case_double("AEIOU"));
    // env.short_insert("Åx", case_double("ZXCVBNM"));
    // env.short_insert("Åy", case_double("AEIOUY"));
    env.short_insert("Åz", str_class("z-aZ-A"));

    env.short_insert(" ", BuiltIn {
        name: "Nop".to_string(),
        func: |_env| Ok(()),
    });
    env.short_insert("\n", BuiltIn {
        name: "Nop".to_string(),
        func: |_env| Ok(()),
    });
    env.short_insert("[", BuiltIn {
        name: "Mark_stack".to_string(),
        func: |env| { env.mark_stack(); Ok(()) },
    });
    env.short_insert("]", BuiltIn {
        name: "Pack".to_string(),
        func: |env| {
            let list = env.pop_until_stack_marker();
            env.push(pd_list(list));
            Ok(())
        },
    });
    env.short_insert("¬", BuiltIn {
        name: "Pack_reverse".to_string(),
        func: |env| {
            let mut list = env.pop_until_stack_marker();
            list.reverse();
            env.push(pd_list(list));
            Ok(())
        },
    });
    env.short_insert("~", BuiltIn {
        name: "Expand_or_eval".to_string(),
        func: |env| {
            match env.pop_result("~ failed")? {
                PdObj::Block(bb) => bb.run(env),
                PdObj::List(ls) => { env.extend_clone(&*ls); Ok(()) }
                _ => Err(PdError::BadArgument("~ can't handle".to_string())),
            }
        },
    });
    env.short_insert("O", BuiltIn {
        name: "Print".to_string(),
        func: |env| {
            let obj = env.pop_result("O failed")?;
            print!("{}", env.to_string(&obj));
            Ok(())
        },
    });
    env.short_insert("P", BuiltIn {
        name: "Print".to_string(),
        func: |env| {
            let obj = env.pop_result("P failed")?;
            println!("{}", env.to_string(&obj));
            Ok(())
        },
    });
    env.short_insert("?", BuiltIn {
        name: "If_else".to_string(),
        func: |env| {
            let else_branch = env.pop_result("If_else else failed")?;
            let if_branch   = env.pop_result("If_else if failed")?;
            let condition   = env.pop_result("If_else condition failed")?;
            if pd_truthy(&condition) {
                apply_on(env, if_branch)
            } else {
                apply_on(env, else_branch)
            }
        },
    });
    env.short_insert("Á", DeepZipBlock {
        func: |a, b| a + b,
        name: "plus".to_string(),
    });
    env.short_insert("Uc", DeepCharToCharBlock {
        func: |a| a.to_uppercase().next().expect("uppercase :("), // FIXME uppercasing chars can produce more than one!
        name: "uppercase".to_string(),
    });
    env.short_insert("Lc", DeepCharToCharBlock {
        func: |a| a.to_lowercase().next().expect("lowercase :("), // FIXME
        name: "lowercase".to_string(),
    });
    env.short_insert("Xc", DeepCharToCharBlock {
        func: |a| {
            if a.is_lowercase() {
                a.to_uppercase().next().expect("swap to uppercase :(")
            } else {
                a.to_lowercase().next().expect("swap to lowercase :(")
            }
        },
        name: "swapcase".to_string(),
    });
}

pub fn simple_eval(code: &str) -> Vec<PdObj> {
    let mut env = Environment::new();
    initialize(&mut env);

    let block = CodeBlock::parse(code);

    match block.run(&mut env) {
        Ok(()) => {}
        Err(e) => { panic!("{:?}", e); }
    }

    env.stack
}
