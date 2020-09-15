#[macro_use] extern crate lazy_static;

use std::cmp::Ordering;
use std::ops::BitAnd;
use std::rc::Rc;
use std::slice::Iter;
use std::fmt::Debug;
use std::mem;
use std::fmt;
use num_iter;
use num::bigint::BigInt;
use num::Integer;
use num_traits::cast::ToPrimitive;
use num_traits::pow::Pow;
use num_traits::identities::Zero;
use std::collections::{HashSet, HashMap};
use std::cell::RefCell;
use std::iter::FromIterator;
use rand;

mod lex;
mod pdnum;
mod pderror;
mod input;
mod slice_util;
mod string_util;
mod vec_util;
mod hoard;
mod char_info;
mod gamma;
use crate::pdnum::{PdNum, PdTotalNum};
use crate::pderror::{PdError, PdResult, PdUnit};
use crate::input::{InputTrigger, ReadValue, EOFReader};
use crate::string_util::{str_class, int_groups, float_groups};
use crate::vec_util as vu;
use crate::hoard::{Hoard, HoardKey};
use crate::lex::AssignType;

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
    fn pop_x(&mut self) {
        self.borrow_x_stack_mut().pop().expect("m8 pop_x");
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
    fn set_zyx(&mut self, z: PdObj, y: PdObj, x: PdObj) {
        let x_stack = self.borrow_x_stack_mut();
        let len = x_stack.len();
        x_stack[len - 3] = z;
        x_stack[len - 2] = y;
        x_stack[len - 1] = x;
    }
    fn pop_yx(&mut self) {
        self.pop_x();
        self.pop_x();
    }

    fn insert(&mut self, name: String, obj: impl Into<PdObj>) {
        let vars = self.borrow_variables();
        vars.insert(name, obj.into());
    }

    fn insert_builtin(&mut self, name: &str, obj: impl Into<PdObj>) {
        let vars = self.borrow_variables();
        let s = name.to_string();
        if vars.contains_key(&s) {
            panic!("dupe key in insert_builtin: {}", s);
        }
        vars.insert(s, obj.into());
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
            PdObj::Hoard(h) => h.borrow().iter().map(|o| self.to_string(o)).collect::<Vec<String>>().join(""),
            PdObj::Block(b) => b.code_repr(),
        }
    }

    fn to_repr_string(&self, obj: &PdObj) -> String {
        match obj {
            PdObj::Num(n) => n.repr(),
            PdObj::String(s) => format!("\"{}\"", &s.iter().collect::<String>()),
            PdObj::List(v) => format!("[{}]", v.iter().map(|o| self.to_repr_string(o)).collect::<Vec<String>>().join(" ")),
            PdObj::Hoard(h) => format!("Hoard({})", h.borrow().iter().map(|o| self.to_repr_string(o)).collect::<Vec<String>>().join(" ")), // FIXME dicts are gone
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

        let ret_result = body(&mut benv);

        let shadow = benv.shadow.expect("Bracketed shadow disappeared!?!?");
        let arity = shadow.arity;
        *self = *(shadow.env);

        Ok((ret_result?, arity))
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
    Hoard(Rc<RefCell<Hoard<PdKey, PdObj>>>),
}

impl fmt::Display for PdObj {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PdObj::Num(n) => write!(formatter, "{}", n),
            PdObj::String(s) => {
                // FIXME
                write!(formatter, "\"")?;
                for c in s.iter() { write!(formatter, "{}", c)?; }
                write!(formatter, "\"")
            }
            PdObj::List(xs) => {
                write!(formatter, "[")?;
                let mut started = false;
                for x in xs.iter() {
                    if started { write!(formatter, " ")?; }
                    started = true;
                    write!(formatter, "{}", x)?;
                }
                write!(formatter, "]")
            }
            PdObj::Block(b) => write!(formatter, "{:?}", b),
            PdObj::Hoard(h) => write!(formatter, "{}", h.borrow()),
        }
    }
}

type PdHoard = Hoard<PdKey, PdObj>;

impl PartialEq for PdObj {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PdObj::Num   (a), PdObj::Num   (b)) => a == b,
            (PdObj::String(a), PdObj::String(b)) => a == b,
            (PdObj::List  (a), PdObj::List  (b)) => a == b,
            (PdObj::Hoard (a), PdObj::Hoard (b)) => Rc::ptr_eq(a, b),
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
impl From<PdHoard> for PdObj {
    fn from(h: PdHoard) -> Self {
        PdObj::Hoard(Rc::new(RefCell::new(h)))
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

impl PdObj {
    fn iverson(x: bool) -> Self {
        PdObj::from(PdNum::iverson(x))
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

fn just_value(obj: &PdObj) -> Option<PdObj> {
    match obj {
        PdObj::Block(_) => None,
        _ => Some(PdObj::clone(obj)),
    }
}
fn just_any(obj: &PdObj) -> Option<PdObj> {
    Some(PdObj::clone(obj))
}
fn seq_as_any(obj: &PdObj) -> Option<PdObj> {
    match obj {
        PdObj::List(_) => Some(PdObj::clone(obj)),
        PdObj::String(_) => Some(PdObj::clone(obj)),
        PdObj::Hoard(_) => Some(PdObj::clone(obj)),
        _ => None,
    }
}
fn just_num(obj: &PdObj) -> Option<Rc<PdNum>> {
    match obj {
        PdObj::Num(n) => Some(Rc::clone(n)),
        _ => None,
    }
}
fn num_to_bigint(obj: &PdObj) -> Option<BigInt> {
    match obj {
        PdObj::Num(n) => n.to_bigint(),
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

fn just_pd_key(obj: &PdObj) -> Option<PdKey> {
    // FIXME this is kinda scary
    pd_key(obj).ok()
}
fn just_hoard(obj: &PdObj) -> Option<Rc<RefCell<PdHoard>>> {
    match obj {
        PdObj::Hoard(h) => Some(Rc::clone(h)),
        _ => None,
    }
}

pub enum PdSeq {
    List(Rc<Vec<PdObj>>),
    String(Rc<Vec<char>>),
    Range(BigInt, BigInt),
}

impl IntoIterator for PdSeq {
    type Item = PdObj;
    type IntoIter = PdIntoIter;

    fn into_iter(self) -> PdIntoIter {
        match self {
            PdSeq::List(v) => PdIntoIter::List(v, 0),
            PdSeq::String(s) => PdIntoIter::String(s, 0),
            PdSeq::Range(a, b) => PdIntoIter::Range(num_iter::range(a, b)),
        }
    }
}

impl<'a> IntoIterator for &'a PdSeq {
    type Item = PdObj;
    type IntoIter = PdIter<'a>;

    fn into_iter(self) -> PdIter<'a> {
        self.iter()
    }
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

impl PdSeqBuildType {
    fn all(it: impl Iterator<Item=PdObj>) -> PdSeqBuildType {
        if it.into_iter().all(|x| match x { PdObj::String(_) => true, _ => false }) {
            PdSeqBuildType::String
        } else {
            PdSeqBuildType::NotString
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

pub enum PdIntoIter {
    List(Rc<Vec<PdObj>>, usize),
    String(Rc<Vec<char>>, usize),
    Range(num_iter::Range<BigInt>),
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

    // poor man's rank-2 type
    fn dual_apply<LF, SF>(&self, lf: LF, sf: SF) -> PdSeq where LF: FnOnce(&Vec<PdObj>) -> Vec<PdObj>, SF: FnOnce(&Vec<char>) -> Vec<char> {
        match self {
            PdSeq::List(v) => PdSeq::List(Rc::new(lf(v))),
            PdSeq::String(s) => PdSeq::String(Rc::new(sf(s))),
            PdSeq::Range(_, _) => PdSeq::List(Rc::new(lf(&self.to_new_vec()))),
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

    fn pythonic_split_right(&self, index: isize) -> PdSeq {
        let uindex = self.pythonic_clamp_slice_index(index);

        match self {
            PdSeq::List(v) => PdSeq::List(Rc::new(v.split_at(uindex).1.to_vec())),
            PdSeq::String(s) => PdSeq::String(Rc::new(s.split_at(uindex).1.to_vec())),
            PdSeq::Range(a, b) => PdSeq::Range(a + uindex, BigInt::clone(b)),
        }
    }

    fn pythonic_mod_slice(&self, modulus: isize) -> PdResult<PdSeq> {
        match self {
            PdSeq::List(v) => Ok(PdSeq::List(Rc::new(slice_util::pythonic_mod_slice(&**v, modulus).ok_or(PdError::IndexError("mod slice sad".to_string()))?.iter().cloned().cloned().collect()))),
            PdSeq::String(s) => Ok(PdSeq::String(Rc::new(slice_util::pythonic_mod_slice(&**s, modulus).ok_or(PdError::IndexError("mod slice sad".to_string()))?.iter().cloned().cloned().collect()))),
            PdSeq::Range(_, _) => Ok(PdSeq::List(Rc::new(slice_util::pythonic_mod_slice(&self.to_new_vec(), modulus).ok_or(PdError::IndexError("mod slice sad".to_string()))?.iter().cloned().cloned().collect()))),
        }
    }

    fn help_rev_copy<T: Clone>(v: &Vec<T>) -> Vec<T> {
        slice_util::rev_copy(v).iter().cloned().cloned().collect()
    }

    fn help_cycle_left<'a, T: Clone>(amt: &'a BigInt) -> impl Fn(&Vec<T>) -> Vec<T> + 'a {
        move |v| {
            if v.is_empty() {
                Vec::new()
            } else {
                let r = amt.mod_floor(&BigInt::from(v.len())).to_usize().expect("mod usize to usize should work");
                let (lh, rh) = v.split_at(r);
                let mut ret = rh.to_vec();
                ret.extend_from_slice(lh);
                ret
            }
        }
    }

    fn rev_copy(&self) -> PdSeq {
        self.dual_apply(PdSeq::help_rev_copy, PdSeq::help_rev_copy)
    }

    fn cycle_left(&self, amt: &BigInt) -> PdSeq {
        self.dual_apply(PdSeq::help_cycle_left(amt), PdSeq::help_cycle_left(amt))
    }

    fn cycle_right(&self, amt: &BigInt) -> PdSeq {
        self.dual_apply(PdSeq::help_cycle_left(&-amt), PdSeq::help_cycle_left(&-amt))
    }

    fn help_repeat<T: Clone>(amt: usize) -> impl Fn(&Vec<T>) -> Vec<T> {
        move |v| {
            std::iter::repeat(v).take(amt).flatten().cloned().collect()
        }
    }

    fn repeat(&self, amt: usize) -> PdSeq {
        self.dual_apply(PdSeq::help_repeat(amt), PdSeq::help_repeat(amt))
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

impl Iterator for PdIntoIter {
    type Item = PdObj;

    fn next(&mut self) -> Option<PdObj> {
        match self {
            PdIntoIter::List(v, i) => {
                let ret = v.get(*i);
                if ret.is_some() { *i += 1; }
                ret.map(PdObj::clone)
            }
            PdIntoIter::String(v, i) => {
                let ret = v.get(*i);
                if ret.is_some() { *i += 1; }
                ret.map(|c| PdObj::from(*c))
            }
            PdIntoIter::Range(rs) => rs.next().map(PdObj::from),
        }
    }
}

fn just_seq(obj: &PdObj) -> Option<PdSeq> {
    match obj {
        PdObj::List(a) => Some(PdSeq::List(Rc::clone(a))),
        PdObj::String(a) => Some(PdSeq::String(Rc::clone(a))),
        PdObj::Hoard(h) => Some(PdSeq::List(Rc::new(h.borrow().iter().cloned().collect()))),
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
fn seq_range_one(obj: &PdObj) -> Option<PdSeq> {
    match obj {
        PdObj::Num(num) => match &**num {
            PdNum::Int(a) => Some(PdSeq::Range(BigInt::from(1), a + 1)),
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
fn always_seq_or_singleton(obj: &PdObj) -> PdSeq {
    match just_seq(obj) {
        Some(v) => v,
        None => PdSeq::List(Rc::new(vec![PdObj::clone(obj)])),
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
fn block_seq_range_case(func: fn(&mut Environment, &Rc<dyn Block>, &PdSeq) -> PdResult<Vec<PdObj>>) -> Rc<dyn Case> {
    Rc::new(BinaryCase { coerce1: just_block, coerce2: seq_range, func })
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

struct DeepNumToNumBlock {
    func: fn(&PdNum) -> PdNum,
    name: String,
}
impl Debug for DeepNumToNumBlock {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(fmt, "DeepNumToNumBlock {{ func: ???, name: {:?} }}", self.name)
    }
}
impl Block for DeepNumToNumBlock {
    fn run(&self, env: &mut Environment) -> PdUnit {
        let a = env.pop_result("deep num to num no stack")?;
        let res = pd_deep_map(&self.func, &a);
        env.push(res);
        Ok(())
    }
    fn code_repr(&self) -> String {
        String::clone(&self.name) + "_deep_num_to_num"
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

struct DeepCharToIntOrZeroBlock {
    func: fn(char) -> i32,
    name: String,
}
impl Debug for DeepCharToIntOrZeroBlock {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(fmt, "DeepCharToIntOrZeroBlock {{ func: ???, name: {:?} }}", self.name)
    }
}
impl Block for DeepCharToIntOrZeroBlock {
    fn run(&self, env: &mut Environment) -> PdUnit {
        let a = env.pop_result("deep char to int|0 no stack")?;
        let res = pd_deep_char_to_int_or_zero(&self.func, &a);
        env.push(res);
        Ok(())
    }
    fn code_repr(&self) -> String {
        String::clone(&self.name) + "_deep_char_to_int_or_zero"
    }
}


fn yx_loop<F>(env: &mut Environment, it: impl Iterator<Item=PdObj>, mut body: F) -> PdUnit where F: FnMut(&mut Environment, usize, PdObj) -> PdUnit {
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

fn pd_loop(env: &mut Environment, func: &Rc<dyn Block>) -> PdUnit {
    yx_loop(env, (0..).map(PdObj::from), |env, _, _| {
        func.run(env)
    })
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

fn pd_map(env: &mut Environment, func: &Rc<dyn Block>, seq: &PdSeq) -> PdResult<PdObj> {
    let bty = seq.build_type();
    let mut acc = Vec::new();
    pd_flatmap_foreach(env, func, |o| { acc.push(o); Ok(()) }, seq.iter())?;
    Ok(pd_build_like(bty, acc))
}

// generically implementing multizip is not super practical, I think
fn pd_zip(env: &mut Environment, func: &Rc<dyn Block>, seq1: &PdSeq, seq2: &PdSeq) -> PdResult<PdObj> {
    let bty = seq1.build_type() & seq2.build_type();
    let mut acc = Vec::new();

    env.push_x(PdObj::from("INTERNAL Z FILLER -- YOU SHOULD NOT SEE THIS".to_string()));
    env.push_x(PdObj::from("INTERNAL Y FILLER -- YOU SHOULD NOT SEE THIS".to_string()));
    env.push_x(PdObj::from("INTERNAL X FILLER -- YOU SHOULD NOT SEE THIS".to_string()));

    let mut ret = Ok(());
    for (i, (obj1, obj2)) in seq1.iter().zip(seq2.iter()).enumerate() {
        env.set_zyx(PdObj::from(i), PdObj::clone(&obj2), PdObj::clone(&obj1));
        match sandbox(env, &func, vec![obj1, obj2]) {
            Ok(objs) => { acc.extend(objs); }
            Err(PdError::Break) => break,
            Err(PdError::Continue) => {}
            Err(e) => { ret = Err(e); break; }
        }
    }

    env.pop_x();
    env.pop_x();
    env.pop_x();

    ret.map(|()| pd_build_like(bty, acc))
}

fn pd_reduce(env: &mut Environment, func: &Rc<dyn Block>, it: PdIter) -> PdResult<PdObj> {
    let mut acc: Option<PdObj> = None;

    for e in it {
        let cur = match acc {
            None => e,
            Some(a) => {
                // pop consumes the sandbox result since we don't use it any more
                sandbox(env, func, vec![a, e])?.pop().ok_or(PdError::EmptyReduceIntermediate)?
            }
        };
        acc = Some(cur);
    }

    acc.ok_or(PdError::BadList("reduce empty list"))
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

fn pd_count_equal(it: PdIter, obj: &PdObj) -> usize {
    let mut count = 0usize;
    for e in it {
        if obj == &e {
            count += 1
        }
    }
    count
}

fn pd_count_by(env: &mut Environment, func: &Rc<dyn Block>, it: PdIter, fty: FilterType) -> PdResult<usize> {
    let mut count = 0usize;
    yx_loop(env, it, |env, _, obj| {
        if fty.accept(sandbox_truthy(env, &func, vec![obj])?) {
            count += 1
        }
        Ok(())
    })?;
    Ok(count)
}

fn pd_find_index_equal(it: PdIter, obj: &PdObj) -> Option<usize> {
    for (i, e) in it.enumerate() {
        if obj == &e {
            return Some(i)
        }
    }
    None
}

fn pd_find_entry(env: &mut Environment, func: &Rc<dyn Block>, it: PdIter, fty: FilterType) -> PdResult<(usize, PdObj)> {
    let mut found = None;
    yx_loop(env, it, |env, i, obj| {
        if fty.accept(sandbox_truthy(env, &func, vec![PdObj::clone(&obj)])?) {
            found = Some((i, obj));
            Err(PdError::Break)
        } else {
            Ok(())
        }
    })?;
    found.ok_or(PdError::EmptyResult("find entry fail".to_string()))
}

struct StringBlock {
    name: &'static str,
    string: Rc<Vec<char>>,
    f: fn(&mut Environment, &Rc<Vec<char>>) -> PdUnit,
}
impl Debug for StringBlock {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(fmt, "StringBlock {{ name: {:?}, string: {:?}, f: ??? }}", self.name, self.string)
    }
}
impl Block for StringBlock {
    fn run(&self, env: &mut Environment) -> PdUnit {
        (self.f)(env, &self.string)
    }
    fn code_repr(&self) -> String {
        format!("{}_{}", self.string.iter().collect::<String>(), self.name)
    }
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
fn pop_seq_range_one_for(env: &mut Environment, name: &'static str) -> PdResult<PdSeq> {
    let opt_seq = seq_range_one(&env.pop_result(format!("{} no stack", name).as_str())?);
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

struct HoardBlock {
    name: &'static str,
    hoard: Rc<RefCell<PdHoard>>,
    f: fn(&mut Environment, &Rc<RefCell<PdHoard>>) -> PdUnit,
}
impl Debug for HoardBlock {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(fmt, "HoardBlock {{ name: {:?}, hoard: {:?}, f: ??? }}", self.name, self.hoard)
    }
}
impl Block for HoardBlock {
    fn run(&self, env: &mut Environment) -> PdUnit {
        (self.f)(env, &self.hoard)
    }
    fn code_repr(&self) -> String {
        // FIXME
        format!("(Hoard)_{}", self.name)
    }
}

#[derive(Debug, PartialEq)]
pub enum RcLeader {
    Lit(PdObj),
    Var(Rc<String>),
    Assign(AssignType),
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
            lex::Leader::Assign(aty) => {
                RcLeader::Assign(aty)
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

fn stb(name: &'static str, s: &Rc<Vec<char>>, f: fn(&mut Environment, &Rc<Vec<char>>) -> PdUnit) -> PdResult<(PdObj, bool)> {
    let string: Rc<Vec<char>> = Rc::clone(s);
    Ok(((PdObj::Block(Rc::new(StringBlock { name, string, f }))), false))
}
fn obb(name: &'static str, bb: &Rc<dyn Block>, f: fn(&mut Environment, &Rc<dyn Block>) -> PdUnit) -> PdResult<(PdObj, bool)> {
    let body: Rc<dyn Block> = Rc::clone(bb);
    Ok(((PdObj::Block(Rc::new(OneBodyBlock { name, body, f }))), false))
}
fn hb(name: &'static str, h: &Rc<RefCell<PdHoard>>, f: fn(&mut Environment, &Rc<RefCell<PdHoard>>) -> PdUnit) -> PdResult<(PdObj, bool)> {
    let hoard: Rc<RefCell<PdHoard>> = Rc::clone(h);
    Ok(((PdObj::Block(Rc::new(HoardBlock { name, hoard, f }))), false))
}


fn simple_interpolate<S>(env: &mut Environment, string: &Vec<char>) -> PdResult<S> where S: FromIterator<char> {
    let slot_count = string.iter().filter(|c| **c == '%').count();
    let mut objs = env.pop_n_result(slot_count, "interpolate stack failed")?.into_iter();
    // somebody needs to own the String while we .chars() it, meh
    let ss: Vec<String> = string.iter().map(|c| {
        match c {
            '%' => env.to_string(&objs.next().expect("interpolate objs count inconsistent")),
            cc => cc.to_string(),
        }
    }).collect();
    Ok(ss.iter().flat_map(|c| c.chars()).collect())
}

fn apply_trailer(outer_env: &mut Environment, obj: &PdObj, trailer0: &lex::Trailer) -> PdResult<(PdObj, bool)> {
    let mut trailer: &str = trailer0.0.as_ref();
    trailer = trailer.strip_prefix('_').unwrap_or(trailer);

    match obj {
        PdObj::Num(n) => match trailer {
            "m" | "minus" => { Ok((PdObj::from(-&**n), false)) }
            "h" | "hundred" => { Ok((PdObj::from(n.mul_const(100)), false)) }
            "k" | "thousand" => { Ok((PdObj::from(n.mul_const(1000)), false)) }
            "p" | "power" => {
                let exponent: PdNum = PdNum::clone(n);
                Ok(((PdObj::Block(Rc::new(DeepBinaryOpBlock { func: |a, b| a.pow_num(b), other: exponent }))), false))
            }
            // TODO maaaaybe look at the sign
            "d" | "digits" => match &**n {
                PdNum::Int(i) => Ok((pd_list(i.to_radix_be(10).1.iter().map(|x| PdObj::from(*x as usize)).collect()), false)),
                _ => Err(PdError::InapplicableTrailer(format!("{} on {}", trailer, obj)))
            }
            "b" | "bits" => match &**n {
                PdNum::Int(i) => Ok((pd_list(i.to_radix_be(10).1.iter().map(|x| PdObj::from(*x as usize)).collect()), false)),
                _ => Err(PdError::InapplicableTrailer(format!("{} on {}", trailer, obj)))
            }
            _ => Err(PdError::InapplicableTrailer(format!("{} on {}", trailer, obj)))
        }
        PdObj::String(s) => match trailer {
            "i" | "interpolate" => stb("interpolate", s, |env, string| {
                let res = simple_interpolate::<Vec<char>>(env, string)?;
                env.push(PdObj::String(Rc::new(res)));
                Ok(())
            }),
            "o" | "interoutput" => stb("interoutput", s, |env, string| {
                print!("{}", simple_interpolate::<String>(env, string)?);
                Ok(())
            }),
            "p" | "interprint" => stb("interprint", s, |env, string| {
                println!("{}", simple_interpolate::<String>(env, string)?);
                Ok(())
            }),
            _ => Err(PdError::InapplicableTrailer(format!("{} on {}", trailer, obj)))
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
            "z" | "zip" => obb("zip", bb, |env, body| {
                let seq2 = pop_seq_range_for(env, "zip(2)")?;
                let seq1 = pop_seq_range_for(env, "zip(1)")?;
                let res = pd_zip(env, body, &seq1, &seq2)?;
                env.push(res);
                Ok(())
            }),
            "l" | "loop" => obb("loop", bb, |env, body| {
                pd_loop(env, body)
            }),
            "m" | "map" => obb("map", bb, |env, body| {
                let seq = pop_seq_range_for(env, "map")?;
                let res = pd_map(env, body, &seq)?;
                env.push(res);
                Ok(())
            }),
            "o" | "onemap" => obb("onemap", bb, |env, body| {
                let seq = pop_seq_range_one_for(env, "onemap")?;
                let res = pd_map(env, body, &seq)?;
                env.push(res);
                Ok(())
            }),
            "f" | "filter" => obb("filter", bb, |env, body| {
                let seq = pop_seq_range_for(env, "filter")?;
                let res = pd_filter(env, body, seq.iter(), FilterType::Filter)?;
                env.push(pd_list(res));
                Ok(())
            }),
            "g" | "get" => obb("get", bb, |env, body| {
                let seq = pop_seq_range_for(env, "get")?;
                let res = pd_find_entry(env, body, seq.iter(), FilterType::Filter)?.1;
                env.push(res);
                Ok(())
            }),
            "i" | "index" => obb("index", bb, |env, body| {
                let seq = pop_seq_range_for(env, "index")?;
                let res = pd_find_entry(env, body, seq.iter(), FilterType::Filter)?.0;
                env.push(PdObj::from(res));
                Ok(())
            }),
            "j" | "reject" => obb("reject", bb, |env, body| {
                let seq = pop_seq_range_for(env, "reject")?;
                let res = pd_filter(env, body, seq.iter(), FilterType::Reject)?;
                env.push(pd_list(res));
                Ok(())
            }),
            "r" | "reduce" => obb("map", bb, |env, body| {
                let seq = pop_seq_range_for(env, "reduce")?;
                let res = pd_reduce(env, body, seq.iter())?;
                env.push(res);
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
                    |_, o| { if pd_truthy(&o) { Ok((false, true)) } else { Ok((true, false)) } }, seq.iter())?;
                env.push(PdObj::iverson(res));
                Ok(())
            }),
            "ê" | "exists" => obb("exists", bb, |env, body| {
                let seq = pop_seq_range_for(env, "exists")?;
                let res = pd_flat_fold_with_short_circuit(env, body, false,
                    |_, o| { if pd_truthy(&o) { Ok((true, true)) } else { Ok((false, false)) } }, seq.iter())?;
                env.push(PdObj::iverson(res));
                Ok(())
            }),
            "ô" | "none" => obb("none", bb, |env, body| {
                let seq = pop_seq_range_for(env, "none")?;
                let res = pd_flat_fold_with_short_circuit(env, body, true,
                    |_, o| { if pd_truthy(&o) { Ok((true, false)) } else { Ok((false, true)) } }, seq.iter())?;
                env.push(PdObj::iverson(res));
                Ok(())
            }),
            "î" | "identical" => obb("identical", bb, |env, body| {
                let seq = pop_seq_range_for(env, "identical")?;
                let res = vu::identical_by(seq, pd_key_projector(env, body))?;
                env.push(PdObj::iverson(res));
                Ok(())
            }),
            "û" | "unique" => obb("unique", bb, |env, body| {
                let seq = pop_seq_range_for(env, "unique")?;
                let res = vu::unique_by(seq, pd_key_projector(env, body))?;
                env.push(PdObj::iverson(res));
                Ok(())
            }),
            "ç" | "count" => obb("count", bb, |env, body| {
                let seq = pop_seq_range_for(env, "count")?;
                let res = pd_count_by(env, body, seq.iter(), FilterType::Filter)?;
                env.push(PdObj::from(res));
                Ok(())
            }),
            "v" | "bindmap" | "vectorize" => obb("bindmap", bb, |env, body| {
                let b = env.pop_result("bindmap nothing to bind")?;
                let bb: Rc<dyn Block> = Rc::new(BindBlock {
                    body: Rc::clone(body),
                    bound_object: b,
                });

                let seq = pop_seq_range_for(env, "bindmap")?;
                let res = pd_map(env, &bb, &seq)?;
                env.push(res);
                Ok(())
            }),
            "y" | "mapbind" => obb("mapbind", bb, |env, body| {
                let seq = pop_seq_range_for(env, "mapbind")?;

                let b = env.pop_result("mapbind nothing to bind")?;
                let bb: Rc<dyn Block> = Rc::new(UnderBindBlock {
                    body: Rc::clone(body),
                    bound_object: b,
                });

                let res = pd_map(env, &bb, &seq)?;
                env.push(res);
                Ok(())
            }),
            "s" | "scan" => obb("scan", bb, |env, body| {
                let seq = pop_seq_range_for(env, "scan")?;
                let res = pd_scan(env, body, seq.iter())?;
                env.push(pd_list(res));
                Ok(())
            }),
            "w" | "deepmap" => obb("deepmap", bb, |env, body| {
                let seq = pop_seq_range_for(env, "deepmap")?.to_rc_pd_obj();
                let res = pd_deep_map_block(env, body, seq)?;
                env.extend(res);
                Ok(())
            }),
            "ø" | "org" | "organize" => obb("organize", bb, |env, body| {
                let seq = pop_seq_range_for(env, "organize")?;
                let res = pd_organize_by(&seq, pd_key_projector(env, body))?;
                env.push(res);
                Ok(())
            }),
            "æ" | "max" => obb("max", bb, |env, body| {
                let seq = pop_seq_range_for(env, "max")?;
                let res = pd_max_by(&seq, pd_key_projector(env, body))?;
                env.push(res);
                Ok(())
            }),
            "œ" | "min" => obb("min", bb, |env, body| {
                let seq = pop_seq_range_for(env, "min")?;
                let res = pd_min_by(&seq, pd_key_projector(env, body))?;
                env.push(res);
                Ok(())
            }),

            _ => Err(PdError::InapplicableTrailer(format!("{} on {}", trailer, obj)))
        }
        PdObj::Hoard(hh) => match trailer {
            "a" | "append" => hb("append", hh, |env, hoard| {
                let obj = env.pop_result("hoard append no stack")?;
                hoard.borrow_mut().push(obj)?;
                Ok(())
            }),
            "b" | "appendleft" => hb("appendleft", hh, |env, hoard| {
                let obj = env.pop_result("hoard append no stack")?;
                hoard.borrow_mut().push_front(obj)?;
                Ok(())
            }),
            "p" | "pop" => hb("pop", hh, |env, hoard| {
                let obj = hoard.borrow_mut().pop()?.ok_or(PdError::BadList("Pop empty list"))?;
                env.push(obj);
                Ok(())
            }),
            "q" | "popleft" => hb("popleft", hh, |env, hoard| {
                let obj = hoard.borrow_mut().pop_front()?.ok_or(PdError::BadList("Pop left empty list"))?;
                env.push(obj);
                Ok(())
            }),
            "x" | "extend" => hb("extend", hh, |env, hoard| {
                let seq = pop_seq_range_for(env, "hoard extend")?;
                hoard.borrow_mut().extend(seq.iter())?;
                Ok(())
            }),

            "g" | "get" => hb("get", hh, |env, hoard| {
                let default_obj = env.pop_result("hoard get default no stack")?;
                let key_obj = env.pop_result("hoard get key no stack")?;
                let key = pd_key(&key_obj)?;
                env.push(hoard.borrow().get(&key).map_or(default_obj, PdObj::clone));
                Ok(())
            }),
            "z" | "getzero" => hb("getzero", hh, |env, hoard| {
                let key_obj = env.pop_result("hoard update key no stack")?;
                let key = pd_key(&key_obj)?;
                env.push(hoard.borrow().get(&key).map_or(PdObj::from(0), PdObj::clone));
                Ok(())
            }),
            "h" | "haskey" => hb("haskey", hh, |env, hoard| {
                let key_obj = env.pop_result("hoard haskey key no stack")?;
                let key = pd_key(&key_obj)?;
                env.push(PdObj::iverson(hoard.borrow().get(&key).is_some()));
                Ok(())
            }),
            "u" | "update" => hb("update", hh, |env, hoard| {
                let val_obj = env.pop_result("hoard update val no stack")?;
                let key_obj = env.pop_result("hoard update key no stack")?;
                let key = pd_key(&key_obj)?;
                hoard.borrow_mut().update(key, val_obj);
                Ok(())
            }),
            "o" | "updateone" => hb("updateone", hh, |env, hoard| {
                let key_obj = env.pop_result("hoard update key no stack")?;
                let key = pd_key(&key_obj)?;
                hoard.borrow_mut().update(key, PdObj::from(1));
                Ok(())
            }),
            "m" | "modify" => hb("modify", hh, |env, hoard| {
                let modifier = env.pop_result("hoard no modifier")?;
                let key_obj = env.pop_result("hoard modifier key no stack")?;
                let key = pd_key(&key_obj)?;
                let value: Option<PdObj> = hoard.borrow().get(&key).map(PdObj::clone);

                let new_value: PdObj = match (value, modifier) {
                    (Some(old), PdObj::Block(b)) => sandbox(env, &b, vec![old])?.pop().ok_or(PdError::EmptyStack("hoard modify produced empty stack".to_string()))?,
                    (None, PdObj::Num(v)) => PdObj::Num(v),
                    (Some(PdObj::Num(v)), PdObj::Num(m)) => PdObj::from(&*m + &*v),
                    _ => Err(PdError::BadArgument("idk how to hoard modify".to_string()))?,
                };

                hoard.borrow_mut().update(key, new_value);
                Ok(())
            }),
            "d" | "delete" => hb("delete", hh, |env, hoard| {
                let key_obj = env.pop_result("hoard update key no stack")?;
                let key = pd_key(&key_obj)?;
                hoard.borrow_mut().delete(&key);
                Ok(())
            }),
            "r" | "replace" => hb("replace", hh, |env, hoard| {
                let obj = env.pop_result("hoard replace no stack")?;
                if let PdObj::Hoard(h) = obj {
                    *hoard.borrow_mut() = Hoard::clone(&h.borrow());
                    Ok(())
                } else if let Some(seq) = seq_num_singleton(&obj) {
                    hoard.borrow_mut().replace_vec(seq.to_new_vec());
                    Ok(())
                } else {
                    Err(PdError::InvalidHoardOperation)
                }
            }),
            "t" | "translate" => hb("translate", hh, |env, hoard| {
                let seq = pop_seq_range_for(env, "hoard translate")?;
                let keys = seq.iter().map(|x| pd_key(&x)).collect::<PdResult<Vec<PdKey>>>()?;

                env.push(pd_list(keys.iter().map(|k| hoard.borrow().get(&k).map_or(PdObj::from(0), PdObj::clone)).collect()));
                Ok(())
            }),

            _ => Err(PdError::InapplicableTrailer(format!("{} on {}", trailer, obj)))
        }
        _ => Err(PdError::InapplicableTrailer(format!("{} on {}", trailer, obj)))
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

        let mut active_assign = None;
        for RcToken(leader, trailer) in &self.1 {
            // println!("{:?} {:?}", leader, trailer);
            match active_assign {
                Some(aty) => {
                    if let RcLeader::Var(s) = leader {
                        let mut var_name = s.to_string();
                        for t in trailer {
                            var_name.push_str(t.0.as_str());
                        }
                        let obj = match aty {
                            AssignType::Peek => env.peek_result("assign")?,
                            AssignType::Pop => env.pop_result("assign")?,
                        };
                        env.insert(var_name, obj);
                    } else {
                        panic!("can't assign to non-var :(");
                    }
                    active_assign = None;
                }
                None => match leader {
                    RcLeader::Lit(obj) => {
                        let (obj, reluctant) = apply_all_trailers(env, PdObj::clone(obj), true, trailer)?;

                        if reluctant {
                            env.push(obj);
                        } else {
                            apply_on(&mut env, obj)?;
                        }
                    }
                    RcLeader::Var(s) => {
                        let (obj, rest) = lookup_and_break_trailers(env, s, trailer).ok_or(PdError::UndefinedVariable(s.to_string() + &trailer.iter().map(|x| x.0.as_str()).collect::<Vec<&str>>().join("")))?;
                        let cobj = PdObj::clone(obj); // borrow checker to drop obj which unborrows env
                        let (obj, reluctant) = apply_all_trailers(env, cobj, false, rest)?;

                        if reluctant {
                            env.push(obj);
                        } else {
                            apply_on(&mut env, obj)?;
                        }
                    }
                    RcLeader::Assign(aty) => {
                        if active_assign.is_some() {
                            panic!("what???");
                        }
                        active_assign = Some(*aty);
                    }
                }
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
        PdObj::Hoard(_)  => { env.push(obj); Ok(()) }
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
        PdObj::Hoard(h) => !h.borrow().is_empty(),
        PdObj::Block(_) => true,
    }
}

fn pd_deep_length(x: &PdObj) -> PdResult<usize> {
    match x {
        PdObj::Num(_) => Ok(1),
        PdObj::String(ss) => Ok(ss.len()),
        PdObj::List(v) => v.iter().map(|x| pd_deep_length(&*x)).sum(),
        PdObj::Hoard(h) => h.borrow().iter().map(|x| pd_deep_length(&*x)).sum(),
        PdObj::Block(_) => Err(PdError::BadArgument("deep length can't block".to_string())),
    }
}

fn pd_deep_sum(x: &PdObj) -> PdResult<PdNum> {
    match x {
        PdObj::Num(n) => Ok((&**n).clone()),
        PdObj::String(ss) => Ok(PdNum::Char(ss.iter().map(|x| BigInt::from(*x as u32)).sum())),
        PdObj::List(v) => v.iter().map(|x| pd_deep_sum(&*x)).sum(),
        PdObj::Hoard(h) => h.borrow().iter().map(|x| pd_deep_sum(&*x)).sum(),
        PdObj::Block(_) => Err(PdError::BadArgument("deep sum can't block".to_string())),
    }
}

fn pd_deep_square_sum(x: &PdObj) -> PdResult<PdNum> {
    match x {
        PdObj::Num(n) => Ok(&**n * &**n),
        PdObj::String(ss) => Ok(PdNum::Char(ss.iter().map(|x| BigInt::from(*x as u32).pow(2u32)).sum())),
        PdObj::List(v) => v.iter().map(|x| pd_deep_square_sum(&*x)).sum(),
        PdObj::Hoard(h) => h.borrow().iter().map(|x| pd_deep_square_sum(&*x)).sum(),
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
        PdObj::Hoard(h) => h.borrow().iter().map(|x| pd_deep_product(&*x)).product(),
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

fn pd_deep_char_to_int_or_zero<F>(f: &F, a: &PdObj) -> PdObj
    where F: Fn(char) -> i32 {

    pd_deep_map(&|num: &PdNum| {
        num.to_char().map_or(PdNum::from(0), |ch| PdNum::from(f(ch)))
    }, a)
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Debug)]
pub enum PdKey {
    Num(PdTotalNum),
    String(Rc<Vec<char>>),
    List(Rc<Vec<Rc<PdKey>>>),
}

impl fmt::Display for PdKey {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PdKey::Num(n) => write!(formatter, "{}", n),
            PdKey::String(s) => {
                // FIXME
                write!(formatter, "\"")?;
                for c in s.iter() { write!(formatter, "{}", c)?; }
                write!(formatter, "\"")
            }
            PdKey::List(xs) => {
                write!(formatter, "[")?;
                let mut started = false;
                for x in xs.iter() {
                    if started { write!(formatter, " ")?; }
                    started = true;
                    write!(formatter, "{}", x)?;
                }
                write!(formatter, "]")
            }
        }
    }
}

impl HoardKey for PdKey {
    fn to_isize(&self) -> Option<isize> {
        match self {
            PdKey::Num(n) => n.to_isize(),
            _ => None,
        }
    }

    fn from_usize(i: usize) -> Self {
        PdKey::Num(PdTotalNum(Rc::new(PdNum::from(i))))
    }
}

fn pd_key(obj: &PdObj) -> PdResult<PdKey> {
    match obj {
        PdObj::Num(x) => Ok(PdKey::Num(PdTotalNum(Rc::clone(x)))),
        PdObj::String(x) => Ok(PdKey::String(Rc::clone(x))),
        PdObj::List(x) => Ok(PdKey::List(Rc::new(x.iter().map(|k| Ok(Rc::new(pd_key(k)?))).collect::<PdResult<Vec<Rc<PdKey>>>>()?))),
        PdObj::Hoard(h) => Ok(PdKey::List(Rc::new(
            h.borrow().iter().map(|k| Ok(Rc::new(pd_key(k)?))).collect::<PdResult<Vec<Rc<PdKey>>>>()?
        ))),
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

fn pd_flatten(a: &PdSeq) -> PdObj {
    let bty = PdSeqBuildType::all(a.iter());
    let v = a.iter().flat_map(|x| always_seq_or_singleton(&x)).collect();
    pd_build_like(bty, v)
}

fn pd_flatten_all(a: &PdObj) -> PdObj {
    match just_seq(a) {
        Some(seq) => {
            let bty = PdSeqBuildType::all(seq.iter());
            let v = seq.iter().map(|x| pd_flatten_all(&x)).flat_map(|x| always_seq_or_singleton(&x)).collect();
            pd_build_like(bty, v)
        }
        None => PdObj::clone(a),
    }
}

fn pd_transpose(seq: &PdSeq) -> Vec<PdObj> {
    let mut acc: Vec<Vec<PdObj>> = Vec::new();
    let bty = PdSeqBuildType::all(seq.iter());

    for row in seq.iter() {
        for (i, obj) in always_seq_or_singleton(&row).iter().enumerate() {
            match acc.get_mut(i) {
                Some(col) => col.push(obj),
                None => acc.push(vec![obj]),
            }
        }
    }

    acc.into_iter().map(|col| pd_build_like(bty, col)).collect()
}

fn pd_seq_intersection(a: &PdSeq, b: &PdSeq) -> PdResult<PdObj> {
    let bty = a.build_type() & b.build_type();
    let acc = vu::intersection(a, b, pd_key)?;
    Ok(pd_build_like(bty, acc))
}

fn pd_seq_union(a: &PdSeq, b: &PdSeq) -> PdResult<PdObj> {
    let bty = a.build_type() & b.build_type();
    let acc = vu::union(a, b, pd_key)?;
    Ok(pd_build_like(bty, acc))
}

fn pd_seq_set_difference(a: &PdSeq, b: &PdSeq) -> PdResult<PdObj> {
    let bty = a.build_type() & b.build_type();
    Ok(pd_build_like(bty, vu::set_difference(a, b, pd_key)?))
}

fn pd_seq_symmetric_difference(a: &PdSeq, b: &PdSeq) -> PdResult<PdObj> {
    let bty = a.build_type() & b.build_type();
    Ok(pd_build_like(bty, vu::symmetric_difference(a, b, pd_key)?))
}

trait PdProj : FnMut(&PdObj) -> PdResult<PdKey> {}
impl<T> PdProj for T where T: FnMut(&PdObj) -> PdResult<PdKey> {}

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

fn pd_organize_by(seq: &PdSeq, proj: impl PdProj) -> PdResult<PdObj> {
    let bty = seq.build_type();
    Ok(pd_list(vu::organize_by(seq, proj)?.into_iter().map(|vs| pd_build_like(bty, vs)).collect()))
}
fn pd_sort_by(seq: &PdSeq, proj: impl PdProj) -> PdResult<PdObj> {
    Ok(pd_build_like(seq.build_type(), vu::sort_by(seq, proj)?))
}
fn pd_max_by(seq: &PdSeq, proj: impl PdProj) -> PdResult<PdObj> {
    vu::max_by(seq, proj).and_then(|x| x.ok_or(PdError::BadList("max of empty")))
}
fn pd_min_by(seq: &PdSeq, proj: impl PdProj) -> PdResult<PdObj> {
    vu::min_by(seq, proj).and_then(|x| x.ok_or(PdError::BadList("min of empty")))
}
fn pd_maxima_by(seq: &PdSeq, proj: impl PdProj) -> PdResult<PdObj> {
    let bty = seq.build_type();
    vu::maxima_by(seq, proj).map(|x| pd_build_like(bty, x))
}
fn pd_minima_by(seq: &PdSeq, proj: impl PdProj) -> PdResult<PdObj> {
    let bty = seq.build_type();
    vu::minima_by(seq, proj).map(|x| pd_build_like(bty, x))
}
fn pd_group_by(seq: &PdSeq, proj: impl PdProj) -> PdResult<PdObj> {
    let groups = vu::group_by(seq, proj)?;
    let bty = seq.build_type();
    Ok(pd_list(groups.into_iter().map(|group| pd_build_like(bty, group)).collect()))
}
fn pd_uniquify_by(seq: &PdSeq, proj: impl PdProj) -> PdResult<PdObj> {
    Ok(pd_build_like(seq.build_type(), vu::uniquify_by(seq, proj)?))
}

fn pd_mul_div_const(env: &mut Environment, obj: &PdObj, mul: usize, div: usize) -> PdResult<Vec<PdObj>> {
    match obj {
        PdObj::Num(n) => {
            if div == 1 {
                Ok(vec![PdObj::from(n.mul_const(mul as i32))])
            } else {
                Ok(vec![PdObj::from(n.mul_div_const(mul as i32, div as i32))])
            }
        }
        PdObj::List(x) => {
            let target_len = x.len() * mul / div;
            Ok(vec![pd_list(x.iter().cycle().take(target_len).cloned().collect())])
        }
        PdObj::String(x) => {
            let target_len = x.len() * mul / div;
            Ok(vec![PdObj::from(x.iter().cycle().take(target_len).cloned().collect::<Vec<char>>())])
        }
        PdObj::Block(b) => {
            if div == 1 {
                pd_xloop(env, b, PdSeq::Range(BigInt::from(0), BigInt::from(mul)).iter())?;
            } else {
                let threshold = (mul as f64) / (div as f64);
                if rand::random::<f64>() < threshold {
                    b.run(env)?;
                }
            }
            Ok(vec![])
        }
        PdObj::Hoard(x) => {
            let h = x.borrow();
            let target_len = h.len() * mul / div;
            Ok(vec![pd_list(h.iter().cycle().take(target_len).cloned().collect())])
        }
    }
}

pub fn initialize(env: &mut Environment) {
    let plus_case = nn_n![a, b, a + b];
    let minus_case = nn_n![a, b, a - b];
    let antiminus_case = nn_n![a, b, b - a];
    let times_case = nn_n![a, b, a * b];
    // TODO: signs...
    let div_case = nn_n![a, b, a / b];
    let mod_case = nn_n![a, b, a % b];
    let intdiv_case = nn_n![a, b, a.div_floor(b)];

    let bitand_case = nn_n![a, b, a & b];
    let bitor_case  = nn_n![a, b, a | b];
    let bitxor_case = nn_n![a, b, a ^ b];
    let gcd_case    = nn_n![a, b, a.gcd(b)];

    let inc_case   = n_n![a, a.add_const( 1)];
    let dec_case   = n_n![a, a.add_const(-1)];
    let inc2_case  = n_n![a, a.add_const( 2)];
    let dec2_case  = n_n![a, a.add_const(-2)];
    let double_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |env, a| pd_mul_div_const(env, a, 2, 1) });
    let frac_14_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |env, a| pd_mul_div_const(env, a, 1, 4) });
    let frac_12_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |env, a| pd_mul_div_const(env, a, 1, 2) });
    let frac_34_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |env, a| pd_mul_div_const(env, a, 3, 4) });

    let ceil_case   = n_n![a, a.ceil()];
    let floor_case  = n_n![a, a.floor()];
    let abs_case    = n_n![a, a.abs()];
    let neg_case    = n_n![a, -a];
    let signum_case = n_n![a, a.signum()];
    let trunc_case  = n_n![a, a.trunc()];
    let equals_one_case = n_n![a, PdNum::iverson(a == &PdNum::from(1))];
    let positive_case = n_n![a, PdNum::iverson(a > &PdNum::from(0))];
    let negative_case = n_n![a, PdNum::iverson(a < &PdNum::from(0))];
    let positive_or_zero_case = n_n![a, PdNum::iverson(a >= &PdNum::from(0))];
    let negative_or_zero_case = n_n![a, PdNum::iverson(a <= &PdNum::from(0))];
    let even_case = n_n![a, PdNum::iverson(a % &PdNum::from(2) == PdNum::from(0))];
    let odd_case  = n_n![a, PdNum::iverson(a % &PdNum::from(2) == PdNum::from(1))];
    let two_to_the_power_of_case = n_n![a, PdNum::from(2).pow_num(a)];
    let factorial_case = n_n![a, a.factorial()];

    let eq_case = nn_n![a, b, PdNum::iverson(a == b)];
    let lt_case = nn_n![a, b, PdNum::iverson(a < b)];
    let gt_case = nn_n![a, b, PdNum::iverson(a > b)];
    let min_case = nn_n![a, b, PdNum::clone(a.min(b))];
    let max_case = nn_n![a, b, PdNum::clone(a.max(b))];

    let min_seq_case = unary_seq_case(|_, seq: &PdSeq| Ok(vec![pd_min_by(seq, pd_key)?]));
    let max_seq_case = unary_seq_case(|_, seq: &PdSeq| Ok(vec![pd_max_by(seq, pd_key)?]));
    let min_seq_by_case: Rc<dyn Case> = block_seq_range_case(|env, block, seq| {
        Ok(vec![pd_min_by(seq, pd_key_projector(env, block))?])
    });
    let max_seq_by_case: Rc<dyn Case> = block_seq_range_case(|env, block, seq| {
        Ok(vec![pd_max_by(seq, pd_key_projector(env, block))?])
    });
    let minima_seq_case = unary_seq_case(|_, seq: &PdSeq| Ok(vec![pd_minima_by(seq, pd_key)?]));
    let maxima_seq_case = unary_seq_case(|_, seq: &PdSeq| Ok(vec![pd_maxima_by(seq, pd_key)?]));
    let minima_seq_by_case: Rc<dyn Case> = block_seq_range_case(|env, block, seq| {
        Ok(vec![pd_minima_by(seq, pd_key_projector(env, block))?])
    });
    let maxima_seq_by_case: Rc<dyn Case> = block_seq_range_case(|env, block, seq| {
        Ok(vec![pd_maxima_by(seq, pd_key_projector(env, block))?])
    });

    let uncons_case = unary_seq_range_case(|_, a| {
        let (x, xs) = a.split_first().ok_or(PdError::BadList("Uncons of empty list"))?;
        Ok(vec![xs, x])
    });
    let hoard_first_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_hoard,
        func: |_, hoard| { Ok(vec![PdObj::clone(hoard.borrow().first().ok_or(PdError::BadList("first of empty hoard"))?)]) },
    });
    let hoard_last_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_hoard,
        func: |_, hoard| { Ok(vec![PdObj::clone(hoard.borrow().last().ok_or(PdError::BadList("last of empty hoard"))?)]) },
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
        let mut arity = 0;
        for case in cases.iter() {
            if case.arity() < arity {
                panic!("cases not sorted: {}", name);
            }
            arity = case.arity();
        }

        env.insert_builtin(name, CasedBuiltIn {
            name: name.to_string(),
            cases,
        });
    };

    let map_case: Rc<dyn Case> = block_seq_range_case(|env, a, b| {
        Ok(vec![pd_map(env, a, b)?])
    });
    let xloop_case: Rc<dyn Case> = block_seq_range_case(|env, a, b| {
        pd_xloop(env, a, b.iter())?; Ok(vec![])
    });

    let filter_case: Rc<dyn Case> = block_seq_range_case(|env, a, b| {
        Ok(vec![pd_list(pd_filter(env, a, b.iter(), FilterType::Filter)?)])
    });
    let filter_indices_case: Rc<dyn Case> = block_seq_range_case(|env, a, b| {
        Ok(vec![pd_list(pd_filter_indices(env, a, b.iter(), FilterType::Filter)?)])
    });
    let reject_case: Rc<dyn Case> = block_seq_range_case(|env, a, b| {
        Ok(vec![pd_list(pd_filter(env, a, b.iter(), FilterType::Reject)?)])
    });
    let reject_indices_case: Rc<dyn Case> = block_seq_range_case(|env, a, b| {
        Ok(vec![pd_list(pd_filter_indices(env, a, b.iter(), FilterType::Reject)?)])
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
    let index_hoard_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_hoard,
        coerce2: just_pd_key,
        func: |_, hoard, key| {
            Ok(vec![PdObj::clone(hoard.borrow().get(key).ok_or(PdError::IndexError("index hoard".to_string()))?)])
        },
    });
    let find_case: Rc<dyn Case> = block_seq_range_case(|env, block, seq| {
        Ok(vec![pd_find_entry(env, block, seq.iter(), FilterType::Filter)?.1])
    });
    let find_not_case: Rc<dyn Case> = block_seq_range_case(|env, block, seq| {
        Ok(vec![pd_find_entry(env, block, seq.iter(), FilterType::Reject)?.1])
    });
    let find_index_case: Rc<dyn Case> = block_seq_range_case(|env, block, seq| {
        Ok(vec![PdObj::from(pd_find_entry(env, block, seq.iter(), FilterType::Filter)?.0)])
    });
    let count_factors_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: num_to_bigint,
        coerce2: num_to_bigint,
        func: |_, haystack, needle| {
            Ok(vec![if needle.is_zero() {
                PdObj::from(0)
            } else if needle == &BigInt::from(1) || needle == &BigInt::from(-1) {
                PdObj::from(f64::INFINITY)
            } else {
                let ret = 0;
                let mut x = BigInt::clone(haystack);
                while (&x % needle).is_zero() {
                    x /= needle;
                }
                PdObj::from(ret)
            }])
        },
    });
    let count_str_str_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_string,
        coerce2: just_string,
        func: |_, haystack, needle| {
            // FIXME slow idk
            let mut acc = 0usize;
            for w in slice_util::sliding_window(haystack, needle.len()) {
                if w == needle.as_slice() {
                    acc += 1;
                }
            }
            Ok(vec![PdObj::from(acc)])
        },
    });
    let find_index_str_str_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_string,
        coerce2: just_string,
        func: |_, haystack, needle| {
            // FIXME slow idk
            for (i, w) in slice_util::sliding_window(haystack, needle.len()).iter().enumerate() {
                if w == &needle.as_slice() {
                    return Ok(vec![PdObj::from(i)]);
                }
            }
            Ok(vec![PdObj::from(-1)])
        },
    });
    let count_equal_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_seq,
        coerce2: just_value,
        func: |_, seq, obj| {
            Ok(vec![PdObj::from(pd_count_equal(seq.iter(), obj))])
        },
    });
    let count_by_case: Rc<dyn Case> = block_seq_range_case(|env, block, seq| {
        Ok(vec![PdObj::from(pd_count_by(env, block, seq.iter(), FilterType::Filter)?)])
    });
    let find_index_equal_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_seq,
        coerce2: just_value,
        func: |_, seq, obj| {
            Ok(vec![pd_find_index_equal(seq.iter(), obj).map_or(PdObj::from(-1), PdObj::from)])
        },
    });
    let hoard_len_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_hoard,
        func: |_, hoard| { Ok(vec![(PdObj::from(hoard.borrow().len()))]) },
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
    let cycle_left_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: seq_range,
        coerce2: num_to_bigint,
        func: |_, seq, num| {
            Ok(vec![seq.cycle_left(num).to_rc_pd_obj()])
        },
    });
    let cycle_right_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: seq_range,
        coerce2: num_to_bigint,
        func: |_, seq, num| {
            Ok(vec![seq.cycle_right(num).to_rc_pd_obj()])
        },
    });
    let cycle_left_one_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: seq_range,
        func: |_, seq| {
            Ok(vec![seq.cycle_left(&BigInt::from(1)).to_rc_pd_obj()])
        },
    });
    let cycle_right_one_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: seq_range,
        func: |_, seq| {
            Ok(vec![seq.cycle_right(&BigInt::from(1)).to_rc_pd_obj()])
        },
    });
    let repeat_seq_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_seq,
        coerce2: num_to_clamped_usize,
        func: |_, a, n| {
            Ok(vec![a.repeat(*n).to_rc_pd_obj()])
        },
    });
    let seq_split_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_seq,
        coerce2: num_to_nn_usize,
        func: |_, seq, size| {
            Ok(vec![pd_list(slice_util::split_slice(seq.to_new_vec().as_slice(), *size, true).iter().map(|s| pd_build_like(seq.build_type(), s.to_vec())).collect())])
        },
    });
    let seq_split_discarding_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_seq,
        coerce2: num_to_nn_usize,
        func: |_, seq, size| {
            Ok(vec![pd_list(slice_util::split_slice(seq.to_new_vec().as_slice(), *size, false).iter().map(|s| pd_build_like(seq.build_type(), s.to_vec())).collect())])
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
    let seq_window_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_seq,
        coerce2: num_to_nn_usize,
        func: |_, seq, size| {
            Ok(vec![pd_list(slice_util::sliding_window(seq.to_new_vec().as_slice(), *size).iter().map(|s| pd_build_like(seq.build_type(), s.to_vec())).collect())])
        },
    });
    let seq_words_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![pd_list(slice_util::split_slice_by_predicate(seq.to_new_vec().as_slice(), |x| {
                if let PdObj::Num(n) = x {
                    if let PdNum::Char(_) = **n {
                        n.to_char().map_or(false, char::is_whitespace)
                    } else {
                        false
                    }
                } else {
                    false
                }
            }, true).iter().map(|s| pd_build_like(seq.build_type(), s.to_vec())).collect())])
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

    let cartesian_product_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: seq_range,
        coerce2: seq_range,
        func: |_, seq1: &PdSeq, seq2: &PdSeq| {
            Ok(vec![pd_list(
                    seq1.iter().map(|e1| pd_list(seq2.iter().map(|e2| pd_list(vec![PdObj::clone(&e1), e2])).collect())).collect())])
        },
    });
    let flat_cartesian_product_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_seq,
        coerce2: just_seq,
        func: |_, seq1: &PdSeq, seq2: &PdSeq| {
            Ok(vec![pd_list(
                    seq1.iter().flat_map(|e1| seq2.iter().map(move |e2| pd_list(vec![PdObj::clone(&e1), e2]))).collect())])
        },
    });
    let square_cartesian_product_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq: &PdSeq| {
            Ok(vec![pd_list(
                    seq.iter().map(|e1| pd_list(seq.iter().map(|e2| pd_list(vec![PdObj::clone(&e1), e2])).collect())).collect())])
        },
    });

    let subsequences_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq: &PdSeq| {
            Ok(vec![pd_list(
                    vu::subsequences(&seq.to_new_vec()).into_iter().map(pd_list).collect())])
        },
    });
    let permutations_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq: &PdSeq| {
            Ok(vec![pd_list(
                    vu::permutations(&seq.to_new_vec()).into_iter().map(pd_list).collect())])
        },
    });

    let flatten_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq: &PdSeq| {
            Ok(vec![pd_flatten(seq)])
        },
    });
    let flatten_all_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: seq_as_any,
        func: |_, obj| {
            Ok(vec![pd_flatten_all(obj)])
        },
    });
    let transpose_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq: &PdSeq| {
            Ok(vec![pd_list(pd_transpose(seq))])
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
    let anti_set_difference_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: seq_range,
        coerce2: seq_range,
        func: |_, seq1, seq2| {
            Ok(vec![pd_seq_set_difference(seq2, seq1)?])
        },
    });
    let symmetric_difference_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: seq_range,
        coerce2: seq_range,
        func: |_, seq1, seq2| {
            Ok(vec![pd_seq_symmetric_difference(seq1, seq2)?])
        },
    });
    let all_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![PdObj::iverson(seq.iter().all(|e| pd_truthy(&e)))])
        },
    });
    let any_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![PdObj::iverson(seq.iter().any(|e| pd_truthy(&e)))])
        },
    });
    let not_all_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![PdObj::iverson(!seq.iter().all(|e| pd_truthy(&e)))])
        },
    });
    let not_any_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![PdObj::iverson(!seq.iter().any(|e| pd_truthy(&e)))])
        },
    });
    let identical_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![PdObj::iverson(vu::identical_by(seq, pd_key)?)])
        },
    });
    let unique_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![PdObj::iverson(vu::unique_by(seq, pd_key)?)])
        },
    });
    let uniquify_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![pd_uniquify_by(seq, pd_key)?])
        },
    });
    let first_duplicate_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![vu::first_duplicate_by(seq, pd_key)?.ok_or(PdError::BadList("no duplicates"))?])
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
    let down_one_range_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_num,
        func: |_, num| {
            let n = num.to_bigint().ok_or(PdError::BadFloat)?;
            let vs = num_iter::range_inclusive(BigInt::from(1), n).rev().map(|x| PdObj::from(num.construct_like_self(x))).collect();
            Ok(vec![pd_list(vs)])
        },
    });
    let map_down_singleton_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![pd_list(seq.iter().map(|e| always_seq_or_singleton(&e).rev_copy().to_rc_pd_obj()).collect())])
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

    let range_len_keep_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_any,
        func: |_, obj| {
            // TODO: should we propagate char-ness?
            let n = match obj {
                PdObj::Num(num) => num.to_bigint().ok_or(PdError::BadFloat)?,
                PdObj::List(lst) => BigInt::from(lst.len()),
                PdObj::String(s) => BigInt::from(s.len()),
                PdObj::Hoard(h) => BigInt::from(h.borrow().len()),
                PdObj::Block(_) => Err(PdError::BadArgument("range len keep got block".to_string()))?,
            };
            let vs = num_iter::range(BigInt::from(0), n).map(|x| PdObj::from(x)).collect();
            Ok(vec![PdObj::clone(obj), PdObj::List(Rc::new(vs))])
        },
    });

    let group_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![pd_group_by(seq, pd_key)?])
        },
    });
    let group_by_case: Rc<dyn Case> = block_seq_range_case(|env, block, seq| {
        Ok(vec![pd_group_by(seq, pd_key_projector(env, block))?])
    });
    // FIXME
    let organize_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![pd_organize_by(seq, pd_key)?])
        },
    });
    let organize_by_case: Rc<dyn Case> = block_seq_range_case(|env, block, seq| {
        Ok(vec![pd_organize_by(seq, pd_key_projector(env, block))?])
    });

    let sort_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![pd_sort_by(seq, pd_key)?])
        },
    });
    let sort_by_case: Rc<dyn Case> = block_seq_range_case(|env, block, seq| {
        Ok(vec![pd_sort_by(seq, pd_key_projector(env, block))?])
    },);

    let is_sorted_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![PdObj::iverson(vu::is_sorted_by(seq, pd_key, |x| x != Ordering::Greater)?)])
        },
    });
    let is_sorted_by_case: Rc<dyn Case> = block_seq_range_case(|env, block, seq| {
        Ok(vec![PdObj::iverson(vu::is_sorted_by(seq, pd_key_projector(env, block), |x| x != Ordering::Greater)?)])
    });

    let is_strictly_increasing_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![PdObj::iverson(vu::is_sorted_by(seq, pd_key, |x| x == Ordering::Less)?)])
        },
    });
    let is_strictly_increasing_by_case: Rc<dyn Case> = block_seq_range_case(|env, block, seq| {
        Ok(vec![PdObj::iverson(vu::is_sorted_by(seq, pd_key_projector(env, block), |x| x == Ordering::Less)?)])
    });
    let is_strictly_decreasing_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_seq,
        func: |_, seq| {
            Ok(vec![PdObj::iverson(vu::is_sorted_by(seq, pd_key, |x| x == Ordering::Greater)?)])
        },
    });
    let is_strictly_decreasing_by_case: Rc<dyn Case> = block_seq_range_case(|env, block, seq| {
        Ok(vec![PdObj::iverson(vu::is_sorted_by(seq, pd_key_projector(env, block), |x| x == Ordering::Greater)?)])
    });

    let just_if_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_any,
        coerce2: just_block,
        func: |env, cond, block| {
            if pd_truthy(&cond) {
                block.run(env)?;
            }
            Ok(vec![])
        },
    });
    let just_unless_case: Rc<dyn Case> = Rc::new(BinaryCase {
        coerce1: just_any,
        coerce2: just_block,
        func: |env, cond, block| {
            if !pd_truthy(&cond) {
                block.run(env)?;
            }
            Ok(vec![])
        },
    });

    add_cases("+", cc![plus_case, cat_list_case, filter_case]);
    add_cases("-", cc![minus_case, set_difference_case, reject_case]);
    add_cases("¯", cc![antiminus_case, anti_set_difference_case]);
    add_cases("*", cc![times_case, repeat_seq_case, flat_cartesian_product_case, xloop_case]);
    add_cases("T", cc![cartesian_product_case]);
    add_cases("/", cc![div_case, seq_split_case, str_split_by_case, seq_split_by_case]);
    add_cases("%", cc![mod_case, mod_slice_case, map_case]);
    add_cases("÷", cc![intdiv_case, seq_split_discarding_case]);
    add_cases("&", cc![bitand_case, intersection_case, just_if_case]);
    add_cases("|", cc![bitor_case, union_case, just_unless_case]);
    add_cases("^", cc![bitxor_case, symmetric_difference_case, find_not_case]);
    add_cases("(", cc![dec_case, uncons_case]);
    add_cases(")", cc![inc_case, unsnoc_case]);
    add_cases("=", cc![index_hoard_case, eq_case, index_case, find_case]);
    add_cases("@", cc![find_index_str_str_case, find_index_equal_case, find_index_case]);
    add_cases("#", cc![count_factors_case, count_str_str_case, count_equal_case, count_by_case]);
    add_cases("<", cc![lt_case, lt_slice_case]);
    add_cases(">", cc![gt_case, ge_slice_case]);
    add_cases("<c", cc![cycle_left_case]);
    add_cases(">c", cc![cycle_right_case]);
    add_cases("<o", cc![cycle_left_one_case]);
    add_cases(">o", cc![cycle_right_one_case]);
    add_cases("<m", cc![min_case]);
    add_cases(">m", cc![max_case]);
    add_cases("Õ", cc![min_case]);
    add_cases("Ã", cc![max_case]);
    add_cases("<r", cc![min_seq_case, min_seq_by_case]);
    add_cases(">r", cc![max_seq_case, max_seq_by_case]);
    add_cases("D", cc![down_case]);
    add_cases("W", cc![seq_words_case, seq_window_case]);
    add_cases("L", cc![abs_case, hoard_len_case, len_case]);
    add_cases("M", cc![neg_case]);
    add_cases("U", cc![signum_case, uniquify_case]);
    add_cases("=g", cc![first_duplicate_case]);
    add_cases("Œ", cc![min_seq_case, min_seq_by_case]);
    add_cases("Æ", cc![max_seq_case, max_seq_by_case]);
    add_cases("Œs", cc![minima_seq_case, minima_seq_by_case]);
    add_cases("Æs", cc![maxima_seq_case, maxima_seq_by_case]);
    add_cases("‹", cc![floor_case, hoard_first_case, first_case]);
    add_cases("›", cc![ceil_case, hoard_last_case, last_case]);
    add_cases("«", cc![dec2_case, butlast_case]);
    add_cases("»", cc![inc2_case, rest_case]);
    add_cases("×", cc![double_case]);
    add_cases("½", cc![frac_12_case]);
    add_cases("¼", cc![frac_14_case]);
    add_cases("¾", cc![frac_34_case]);
    add_cases("²", cc![square_case, square_cartesian_product_case]);
    add_cases("¡", cc![factorial_case, permutations_case]);
    add_cases("!p", cc![factorial_case, permutations_case]);
    add_cases("¿", cc![two_to_the_power_of_case, subsequences_case]);
    add_cases("Ss", cc![two_to_the_power_of_case, subsequences_case]);
    add_cases("™", cc![transpose_case]);
    add_cases(" r", cc![space_join_case]);
    add_cases(",", cc![range_case, zip_range_case, filter_indices_case]);
    add_cases("J", cc![one_range_case, zip_one_range_case, reject_indices_case]);
    add_cases("Ð", cc![down_one_range_case, map_down_singleton_case]);
    add_cases("…", cc![flatten_all_case, to_range_case]);
    add_cases("¨", cc![flatten_case, til_range_case]);
    add_cases("´", cc![range_len_keep_case]);
    add_cases("To", cc![to_range_case]);
    add_cases("Tl", cc![til_range_case]);

    add_cases("G", cc![group_case, gcd_case, group_by_case]);
    add_cases("Ø", cc![organize_case, organize_by_case]);
    add_cases("$", cc![sort_case, sort_by_case]);
    add_cases("$p", cc![is_sorted_case, is_sorted_by_case]);
    add_cases("<p", cc![is_strictly_increasing_case, is_strictly_increasing_by_case]);
    add_cases(">p", cc![is_strictly_decreasing_case, is_strictly_decreasing_by_case]);
    add_cases("Â", cc![positive_case, all_case]);
    add_cases("Ê", cc![even_case, any_case]);
    add_cases("Î", cc![equals_one_case, identical_case]);
    add_cases("Ô", cc![odd_case, not_any_case]);
    add_cases("Û", cc![negative_case, unique_case]);
    add_cases("Al", cc![all_case]);
    add_cases("Ay", cc![any_case]);
    add_cases("Na", cc![not_all_case]);
    add_cases("Ne", cc![not_any_case]);
    add_cases("=p", cc![identical_case]);
    add_cases("Ev", cc![even_case]);
    add_cases("Od", cc![odd_case]);
    add_cases("+p", cc![positive_case]);
    add_cases("-p", cc![negative_case]);
    add_cases("+o", cc![positive_or_zero_case]);
    add_cases("-o", cc![negative_or_zero_case]);

    add_cases(":",   vec![juggle!(a -> a, a)]);
    add_cases(":p",  vec![juggle!(a, b -> a, b, a, b)]);
    add_cases("¦",   vec![juggle!(a, b -> a, b, a, b)]);
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

    let pop_if_true_case:  Rc<dyn Case> = Rc::new(UnaryAnyCase  { func: |_, a| Ok(vec![pd_list(if pd_truthy(a) { vec![] } else { vec![PdObj::clone(a)] })]) });
    let pop_if_false_case: Rc<dyn Case> = Rc::new(UnaryAnyCase  { func: |_, a| Ok(vec![pd_list(if pd_truthy(a) { vec![PdObj::clone(a)] } else { vec![] })]) });
    let pop_if_case:       Rc<dyn Case> = Rc::new(BinaryAnyCase { func: |_, a, b| Ok(vec![pd_list(if pd_truthy(b) { vec![] } else { vec![PdObj::clone(a)] })]) });
    let pop_if_not_case:   Rc<dyn Case> = Rc::new(BinaryAnyCase { func: |_, a, b| Ok(vec![pd_list(if pd_truthy(b) { vec![PdObj::clone(a)] } else { vec![] })]) });
    add_cases(";i",  vec![pop_if_case]);
    add_cases(";n",  vec![pop_if_not_case]);
    add_cases(";t",  vec![pop_if_true_case]);
    add_cases(";f",  vec![pop_if_false_case]);

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

    let to_float_case: Rc<dyn Case> = n_n![a, a.to_f64().expect("can't to_float")];
    let string_to_float_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_string,
        func: |_, s: &Rc<Vec<char>>| Ok(vec![PdObj::from(s.iter().collect::<String>().parse::<f64>().map_err(|_| PdError::BadParse)?)]),
    });
    let fixed_point_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_block,
        func: |env, block| Ok(vec![pd_iterate(env, block)?.1]),
    });
    add_cases("F", cc![to_float_case, string_to_float_case, fixed_point_case]);

    let float_groups_case: Rc<dyn Case> = Rc::new(UnaryCase {
        coerce: just_string,
        func: |_, s: &Rc<Vec<char>>| Ok(vec![pd_list(float_groups(&s.iter().collect::<String>()).map(|i| PdObj::from(i)).collect())]),
    });
    add_cases("Fg", cc![float_groups_case]);

    let to_string_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |env, a| Ok(vec![PdObj::from(env.to_string(a))]) });
    add_cases("S", cc![to_string_case]);

    let to_repr_string_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |env, a| Ok(vec![PdObj::from(env.to_repr_string(a))]) });
    add_cases("`", cc![to_repr_string_case]);

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

    env.insert_builtin("H", Hoard::new());
    env.insert_builtin("•", Hoard::new());

    // env.variables.insert("X".to_string(), (PdObj::Int(3.to_bigint().unwrap())));
    env.insert_builtin("N", '\n');
    env.insert_builtin("A", 10);
    env.insert_builtin("¹", 11);
    env.insert_builtin("∅", 0);
    env.insert_builtin("α", 1);
    env.insert_builtin("Ep", 1e-9);

    env.insert_builtin("Da", str_class("0-9"));
    env.insert_builtin("Ua", str_class("A-Z"));
    env.insert_builtin("La", str_class("a-z"));
    env.insert_builtin("Aa", str_class("A-Za-z"));

    // # Non-breaking space (U+00A0)
    env.insert_builtin("\u{a0}", ' ');
    env.insert_builtin("␣", ' ');

    env.insert_builtin("Å", str_class("A-Z"));
    env.insert_builtin("Åa", str_class("a-zA-Z"));
    // env.insert_builtin("Åb", case_double("BCDFGHJKLMNPQRSTVWXZ"));
    // env.insert_builtin("Åc", case_double("BCDFGHJKLMNPQRSTVWXYZ"));
    env.insert_builtin("Åd", str_class("9-0"));
    env.insert_builtin("Åf", str_class("A-Za-z0-9+/"));
    env.insert_builtin("Åh", str_class("0-9A-F"));
    env.insert_builtin("Åi", str_class("A-Za-z0-9_"));
    env.insert_builtin("Åj", str_class("a-zA-Z0-9_"));
    env.insert_builtin("Ål", str_class("z-a"));
    env.insert_builtin("Åm", "()<>[]{}");
    env.insert_builtin("Åp", str_class(" -~"));
    // env.insert_builtin("Åq", case_double("QWERTYUIOP"));
    // env.insert_builtin("Ås", case_double("ASDFGHJKL"));
    env.insert_builtin("Åt", str_class("0-9A-Z"));
    env.insert_builtin("Åu", str_class("Z-A"));
    // env.insert_builtin("Åv", case_double("AEIOU"));
    // env.insert_builtin("Åx", case_double("ZXCVBNM"));
    // env.insert_builtin("Åy", case_double("AEIOUY"));
    env.insert_builtin("Åz", str_class("z-aZ-A"));

    env.insert_builtin(" ", BuiltIn {
        name: "Nop".to_string(),
        func: |_env| Ok(()),
    });
    env.insert_builtin("\n", BuiltIn {
        name: "Nop".to_string(),
        func: |_env| Ok(()),
    });
    env.insert_builtin("[", BuiltIn {
        name: "Mark_stack".to_string(),
        func: |env| { env.mark_stack(); Ok(()) },
    });
    env.insert_builtin("]", BuiltIn {
        name: "Pack".to_string(),
        func: |env| {
            let list = env.pop_until_stack_marker();
            env.push(pd_list(list));
            Ok(())
        },
    });
    env.insert_builtin("¬", BuiltIn {
        name: "Pack_reverse".to_string(),
        func: |env| {
            let mut list = env.pop_until_stack_marker();
            list.reverse();
            env.push(pd_list(list));
            Ok(())
        },
    });
    env.insert_builtin("~", BuiltIn {
        name: "Expand_or_eval".to_string(),
        func: |env| {
            match env.pop_result("~ failed")? {
                PdObj::Block(bb) => bb.run(env),
                PdObj::List(ls) => { env.extend_clone(&*ls); Ok(()) }
                _ => Err(PdError::BadArgument("~ can't handle".to_string())),
            }
        },
    });
    env.insert_builtin("O", BuiltIn {
        name: "Print".to_string(),
        func: |env| {
            let obj = env.pop_result("O failed")?;
            print!("{}", env.to_string(&obj));
            Ok(())
        },
    });
    env.insert_builtin("P", BuiltIn {
        name: "Print".to_string(),
        func: |env| {
            let obj = env.pop_result("P failed")?;
            println!("{}", env.to_string(&obj));
            Ok(())
        },
    });
    env.insert_builtin("V", BuiltIn {
        name: "Input".to_string(),
        func: |env| {
            let obj = env.run_stack_trigger().ok_or(PdError::InputError)?;
            env.push(obj);
            Ok(())
        },
    });
    env.insert_builtin("Q", BuiltIn {
        name: "Break".to_string(),
        func: |_| Err(PdError::Break),
    });
    env.insert_builtin("K", BuiltIn {
        name: "Continue".to_string(),
        func: |_| Err(PdError::Continue),
    });
    env.insert_builtin("?", BuiltIn {
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
    env.insert_builtin("Á", DeepZipBlock {
        func: |a, b| a + b,
        name: "plus".to_string(),
    });
    env.insert_builtin("À", DeepZipBlock {
        func: |a, b| a - b,
        name: "minus".to_string(),
    });
    env.insert_builtin("È", DeepNumToNumBlock {
        func: |a| a * a,
        name: "square".to_string(),
    });
    env.insert_builtin("É", DeepNumToNumBlock {
        func: |a| PdNum::from(2).pow_num(a),
        name: "pow2".to_string(),
    });
    env.insert_builtin("Ì", DeepNumToNumBlock {
        func: |a| -a,
        name: "negate".to_string(),
    });
    env.insert_builtin("Í", DeepNumToNumBlock {
        func: |a| &PdNum::from(1)/a,
        name: "invert".to_string(),
    });
    env.insert_builtin("Ò", DeepZipBlock {
        func: |a, b| a * b,
        name: "times".to_string(),
    });
    env.insert_builtin("Ó", DeepZipBlock {
        func: |a, b| a / b,
        name: "divide".to_string(),
    });
    env.insert_builtin("Ú", DeepZipBlock {
        func: |a, b| a % b,
        name: "mod".to_string(),
    });

    env.insert_builtin("±", DeepZipBlock {
        func: |a, b| (a - b).abs(),
        name: "absdiff".to_string(),
    });

    macro_rules! forward_f64 {
        ($name:expr, $fname:ident) => {
            env.insert_builtin($name, DeepNumToNumBlock {
                func: |a| a.through_float(f64::$fname),
                name: stringify!($fname).to_string(),
            });
        }
    }
    forward_f64!("Sn", sin);
    forward_f64!("Cs", cos);
    forward_f64!("Tn", tan);
    forward_f64!("As", asin);
    forward_f64!("Ac", acos);
    forward_f64!("At", atan);
    env.insert_builtin("Sc", DeepNumToNumBlock {
        func: |a| a.through_float(|f| 1.0/f.cos()),
        name: "Sec".to_string(),
    });
    env.insert_builtin("Cc", DeepNumToNumBlock {
        func: |a| a.through_float(|f| 1.0/f.sin()),
        name: "Csc".to_string(),
    });
    env.insert_builtin("Ct", DeepNumToNumBlock {
        func: |a| a.through_float(|f| 1.0/f.tan()),
        name: "Cot".to_string(),
    });
    forward_f64!("Ef", exp);
    forward_f64!("Ln", ln);
    forward_f64!("Lt", log10);
    forward_f64!("Lg", log2);

    env.insert_builtin("Uc", DeepCharToCharBlock {
        func: |a| a.to_uppercase().next().expect("uppercase :("), // FIXME uppercasing chars can produce more than one!
        name: "uppercase".to_string(),
    });
    env.insert_builtin("Lc", DeepCharToCharBlock {
        func: |a| a.to_lowercase().next().expect("lowercase :("), // FIXME
        name: "lowercase".to_string(),
    });
    env.insert_builtin("Xc", DeepCharToCharBlock {
        func: |a| {
            if a.is_lowercase() {
                a.to_uppercase().next().expect("swap to uppercase :(")
            } else {
                a.to_lowercase().next().expect("swap to lowercase :(")
            }
        },
        name: "swapcase".to_string(),
    });
    env.insert_builtin("Mc", DeepCharToCharBlock {
        func: |a| *char_info::MATCHING_MAP.get(&a).unwrap_or(&a),
        name: "matching_char".to_string(),
    });
    env.insert_builtin("Vc", DeepCharToIntOrZeroBlock {
        func: |a| *char_info::VALUE_MAP.get(&a).unwrap_or(&0),
        name: "value_char".to_string(),
    });
    env.insert_builtin("Nc", DeepCharToIntOrZeroBlock {
        func: |a| *char_info::NEST_MAP.get(&a).unwrap_or(&0),
        name: "nest_char".to_string(),
    });

    env.insert_builtin("·", BuiltIn {
        name: "Assign_bullet".to_string(),
        func: |env| {
            let obj = env.peek_result("Assign_bullet failed")?;
            env.insert("•".to_string(), obj);
            Ok(())
        },
    });
    env.insert_builtin("–", BuiltIn {
        name: "Assign_bullet_destructive".to_string(),
        func: |env| {
            let obj = env.pop_result("Assign_bullet_destructive failed")?;
            env.insert("•".to_string(), obj);
            Ok(())
        },
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
