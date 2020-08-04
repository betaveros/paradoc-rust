#[macro_use] extern crate lazy_static;

use std::cmp::Ordering;
use std::rc::Rc;
use std::slice::Iter;
use std::fmt::Debug;
use std::mem;
use num_iter;
use num::bigint::BigInt;
use num_traits::pow::Pow;
use std::collections::HashMap;

mod lex;
mod pdnum;
use crate::pdnum::PdNum;

#[derive(Debug)]
pub struct Environment {
    stack: Vec<Rc<PdObj>>,
    x_stack: Vec<Rc<PdObj>>,
    variables: HashMap<String, Rc<PdObj>>,
    marker_stack: Vec<usize>,
    // hmmm...
    shadow: Option<ShadowState>,
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
                None => None
            }
        }
        // TODO: stack trigger
    }
    fn pop_or_panic(&mut self, panic_msg: &'static str) -> Rc<PdObj> {
        match self.pop() {
            None => { panic!(panic_msg); }
            Some(x) => x
        }
    }
    fn pop_n_or_panic(&mut self, n: usize, panic_msg: &'static str) -> Vec<Rc<PdObj>> {
        let mut ret: Vec<Rc<PdObj>> = Vec::new();
        for _ in 0..n {
            ret.push(self.pop_or_panic(panic_msg));
        }
        ret.reverse();
        ret
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

    fn pop_until_stack_marker(&mut self) -> Vec<Rc<PdObj>> {
        match self.pop_stack_marker() {
            Some(marker) => {
                self.stack.split_off(marker) // this is way too perfect
            }
            None => {
                panic!("popping TODO this will do stack trigger stuff");
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
                println!("len is {}", self.x_stack.len());
                match name {
                    "X" => self.x_stack.get(self.x_stack.len() - 1),
                    "Y" => self.x_stack.get(self.x_stack.len() - 2),
                    "Z" => self.x_stack.get(self.x_stack.len() - 3),
                    _ => self.variables.get(name)
                }
            }
        }
    }

    fn new() -> Environment {
        Environment {
            stack: Vec::new(),
            x_stack: Vec::new(),
            variables: HashMap::new(),
            marker_stack: Vec::new(),
            shadow: None,
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

    fn run_on_bracketed_shadow<T>(&mut self, shadow_type: ShadowType, body: impl FnOnce(&mut Environment) -> T) -> T {
        self.run_on_bracketed_shadow_with_arity(shadow_type, body).0
    }

    fn run_on_bracketed_shadow_with_arity<T>(&mut self, shadow_type: ShadowType, body: impl FnOnce(&mut Environment) -> T) -> (T, usize) {
        let env = mem::replace(self, Environment::new());

        let mut benv = Environment {
            stack: Vec::new(),
            x_stack: Vec::new(),
            variables: HashMap::new(),
            marker_stack: vec![0],
            shadow: Some(ShadowState { env: Box::new(env), arity: 0, shadow_type }),
        };

        let ret = body(&mut benv);

        let shadow = benv.shadow.expect("Bracketed shadow disappeared!?!?");
        let arity = shadow.arity;
        mem::replace(self, *(shadow.env));

        (ret, arity)
    }
}

pub trait Block : Debug {
    fn run(&self, env: &mut Environment) -> ();
    fn code_repr(&self) -> String;
}

fn sandbox(env: &mut Environment, func: &Rc<dyn Block>, args: Vec<Rc<PdObj>>) -> Vec<Rc<PdObj>> {
    env.run_on_bracketed_shadow(ShadowType::Normal, |inner| {
        inner.extend(args);
        func.run(inner);
        inner.take_stack()
    })
}
#[derive(Debug)]
pub enum PdObj {
    Num(PdNum),
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

macro_rules! forward_from {
    ($ty:ident) => {
        impl From<$ty> for PdObj {
            fn from(n: $ty) -> Self { PdObj::Num(PdNum::from(n)) }
        }
    }
}

forward_from!(BigInt);
forward_from!(char);
forward_from!(i32);
forward_from!(f64);
forward_from!(usize);

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

struct BuiltIn {
    name: String,
    func: fn(&mut Environment) -> (),
}

impl Debug for BuiltIn {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(fmt, "BuiltIn {{ name: {:?}, func: ??? }}", self.name)
    }
}
impl Block for BuiltIn {
    fn run(&self, env: &mut Environment) {
        (self.func)(env);
    }
    fn code_repr(&self) -> String {
        self.name.clone()
    }
}

// Yeah, so we do want to take args because of interactions with shadow stacks
trait Case {
    fn arity(&self) -> usize;
    fn maybe_run_noncommutatively(&self, env: &mut Environment, args: &Vec<Rc<PdObj>>) -> Option<Vec<Rc<PdObj>>>;
}

struct UnaryAnyCase {
    func: fn(&mut Environment, &Rc<PdObj>) -> Vec<Rc<PdObj>>,
}
impl Case for UnaryAnyCase {
    fn arity(&self) -> usize { 1 }
    fn maybe_run_noncommutatively(&self, env: &mut Environment, args: &Vec<Rc<PdObj>>) -> Option<Vec<Rc<PdObj>>> {
        Some((self.func)(env, &args[0]))
    }
}

struct BinaryAnyCase {
    func: fn(&mut Environment, &Rc<PdObj>, &Rc<PdObj>) -> Vec<Rc<PdObj>>,
}
impl Case for BinaryAnyCase {
    fn arity(&self) -> usize { 2 }
    fn maybe_run_noncommutatively(&self, env: &mut Environment, args: &Vec<Rc<PdObj>>) -> Option<Vec<Rc<PdObj>>> {
        Some((self.func)(env, &args[0], &args[1]))
    }
}

struct TernaryAnyCase {
    func: fn(&mut Environment, &Rc<PdObj>, &Rc<PdObj>, &Rc<PdObj>) -> Vec<Rc<PdObj>>,
}
impl Case for TernaryAnyCase {
    fn arity(&self) -> usize { 3 }
    fn maybe_run_noncommutatively(&self, env: &mut Environment, args: &Vec<Rc<PdObj>>) -> Option<Vec<Rc<PdObj>>> {
        Some((self.func)(env, &args[0], &args[1], &args[2]))
    }
}

fn just_num(obj: &PdObj) -> Option<PdNum> {
    match obj {
        PdObj::Num(n) => Some(n.clone()),
        _ => None,
    }
}

pub enum PdSeq {
    List(Rc<Vec<Rc<PdObj>>>),
    String(Rc<Vec<char>>),
    Range(BigInt, BigInt),
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

    // TODO same
    fn first(&self) -> Option<Rc<PdObj>> {
        match self {
            PdSeq::List(v) => v.first().map(Rc::clone),
            PdSeq::String(s) => s.first().map(|x| Rc::new(PdObj::from(*x))),
            PdSeq::Range(a, b) => if a < b { Some(Rc::new(PdObj::from(BigInt::clone(a)))) } else { None },
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

    fn last(&self) -> Option<Rc<PdObj>> {
        match self {
            PdSeq::List(v) => v.last().map(Rc::clone),
            PdSeq::String(s) => s.last().map(|x| Rc::new(PdObj::from(*x))),
            PdSeq::Range(a, b) => if a < b { Some(Rc::new(PdObj::from(b - 1))) } else { None },
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
fn seq_range(obj: &PdObj) -> Option<PdSeq> {
    match obj {
        PdObj::Num(PdNum::Int(a)) => Some(PdSeq::Range(BigInt::from(0), BigInt::clone(a))),
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
    func: fn(&mut Environment, &PdNum) -> Vec<Rc<PdObj>>,
}
impl Case for UnaryNumCase {
    fn arity(&self) -> usize { 1 }
    fn maybe_run_noncommutatively(&self, env: &mut Environment, args: &Vec<Rc<PdObj>>) -> Option<Vec<Rc<PdObj>>> {
        match &*args[0] {
            PdObj::Num(ai) => {
                let ai2 = ai.clone();
                Some((self.func)(env, &ai2))
            }
            _ => None
        }
    }
}

struct UnaryCase<T> {
    coerce: fn(&PdObj) -> Option<T>,
    func: fn(&mut Environment, &T) -> Vec<Rc<PdObj>>,
}
impl<T> Case for UnaryCase<T> {
    fn arity(&self) -> usize { 1 }
    fn maybe_run_noncommutatively(&self, env: &mut Environment, args: &Vec<Rc<PdObj>>) -> Option<Vec<Rc<PdObj>>> {
        match (self.coerce)(&*args[0]) {
            Some(a) => {
                Some((self.func)(env, &a))
            }
            _ => None
        }
    }
}
fn unary_num_case(func: fn(&mut Environment, &PdNum) -> Vec<Rc<PdObj>>) -> Rc<dyn Case> {
    Rc::new(UnaryCase { coerce: just_num, func })
}

struct BinaryCase<T1, T2> {
    coerce1: fn(&PdObj) -> Option<T1>,
    coerce2: fn(&PdObj) -> Option<T2>,
    func: fn(&mut Environment, &T1, &T2) -> Vec<Rc<PdObj>>,
}
impl<T1, T2> Case for BinaryCase<T1, T2> {
    fn arity(&self) -> usize { 2 }
    fn maybe_run_noncommutatively(&self, env: &mut Environment, args: &Vec<Rc<PdObj>>) -> Option<Vec<Rc<PdObj>>> {
        match ((self.coerce1)(&*args[0]), (self.coerce2)(&*args[1])) {
            (Some(a), Some(b)) => {
                Some((self.func)(env, &a, &b))
            }
            _ => None
        }
    }
}
fn binary_num_case(func: fn(&mut Environment, &PdNum, &PdNum) -> Vec<Rc<PdObj>>) -> Rc<dyn Case> {
    Rc::new(BinaryCase { coerce1: just_num, coerce2: just_num, func })
}
fn unary_seq_range_case(func: fn(&mut Environment, &PdSeq) -> Vec<Rc<PdObj>>) -> Rc<dyn Case> {
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
    fn run(&self, env: &mut Environment) {
        let mut done = false;
        let mut accumulated_args: Vec<Rc<PdObj>> = Vec::new();
        for case in &self.cases {
            while accumulated_args.len() < case.arity() {
                match env.pop() {
                    Some(arg) => {
                        accumulated_args.insert(0, arg);
                    }
                    None => {
                        panic!("Ran out of arguments on stack");
                    }
                }
            }
            match case.maybe_run_noncommutatively(env, &accumulated_args) {
                Some(res) => {
                    env.stack.extend(res);
                    done = true;
                    break
                }
                None => {
                    if accumulated_args.len() >= 2 {
                        let len = accumulated_args.len();
                        accumulated_args.swap(len - 1, len - 2);
                        match case.maybe_run_noncommutatively(env, &accumulated_args) {
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
        if !done {
            panic!("No cases of {} applied!", self.name);
        }
    }
    fn code_repr(&self) -> String {
        self.name.clone()
    }
}

#[derive(Debug)]
struct EachBlock {
    body: Rc<dyn Block>,
}
impl Block for EachBlock {
    fn run(&self, env: &mut Environment) {
        // TODO: literally everything
        // Extract into sandbox; push x-stack; handle continue/breaks
        match seq_range(&*env.pop_or_panic("each no stack")) {
            Some(vec) => {
                for obj in vec.iter() {
                    env.push(Rc::clone(&obj));
                    self.body.run(env);
                }
            }
            _ => { panic!("each failed") }
        }
    }
    fn code_repr(&self) -> String {
        self.body.code_repr() + "_each"
    }
}

// TODO: handle continue/break (have fun!)
fn pd_map(env: &mut Environment, func: &Rc<dyn Block>, it: PdIter) -> Vec<Rc<PdObj>> {
    env.push_yx();
    let res = it.enumerate().flat_map(|(i, obj)| {
        env.set_yx(Rc::new(PdObj::from(i)), Rc::clone(&obj));
        sandbox(env, &func, vec![obj])
    }).collect();
    env.pop_yx();
    res
}

#[derive(Debug)]
struct MapBlock {
    body: Rc<dyn Block>,
}
impl Block for MapBlock {
    fn run(&self, env: &mut Environment) {
        match seq_range(&*env.pop_or_panic("map no stack")) {
            Some(vec) => {
                let res = pd_map(env, &self.body, vec.iter());
                env.push(Rc::new(PdObj::List(Rc::new(res))));
            }
            _ => { panic!("map coercion failed") }
        }
    }
    fn code_repr(&self) -> String {
        self.body.code_repr() + "_map"
    }
}

#[derive(Debug)]
struct UnderBlock {
    body: Rc<dyn Block>,
}
impl Block for UnderBlock {
    fn run(&self, env: &mut Environment) {
        let obj = env.pop_or_panic("under no stack");
        self.body.run(env);
        env.push(obj);
    }
    fn code_repr(&self) -> String {
        self.body.code_repr() + "_under"
    }
}

#[derive(Debug)]
struct KeepBlock {
    body: Rc<dyn Block>,
}
impl Block for KeepBlock {
    fn run(&self, env: &mut Environment) {
        let res = env.run_on_bracketed_shadow(ShadowType::Keep, |inner| {
            self.body.run(inner);
            inner.take_stack()
        });
        env.extend(res);
    }
    fn code_repr(&self) -> String {
        self.body.code_repr() + "_keep"
    }
}

#[derive(Debug)]
struct KeepUnderBlock {
    body: Rc<dyn Block>,
}
impl Block for KeepUnderBlock {
    fn run(&self, env: &mut Environment) {
        let (res, arity) = env.run_on_bracketed_shadow_with_arity(ShadowType::Keep, |inner| {
            self.body.run(inner);
            inner.take_stack()
        });
        let temp = env.pop_n_or_panic(arity, "keepunder stack failed");
        env.extend(res);
        env.extend(temp);
    }
    fn code_repr(&self) -> String {
        self.body.code_repr() + "_keepunder"
    }
}

#[derive(Debug, PartialEq)]
pub enum RcLeader {
    Lit(Rc<PdObj>),
    Var(Rc<String>),
}

#[derive(Debug)]
struct CodeBlock(Vec<RcToken>);

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
            lex::Leader::Block(b) => {
                RcLeader::Lit(Rc::new(PdObj::Block(Rc::new(CodeBlock(rcify(*b))))))
            }
            lex::Leader::Var(s) => {
                RcLeader::Var(Rc::new(s))
            }
        };
        RcToken(rcleader, trailer)
    }).collect()
}

fn apply_trailer(obj: &Rc<PdObj>, trailer: &lex::Trailer) -> Option<(Rc<PdObj>, bool)> {
    match &**obj {
        PdObj::Block(bb) => match trailer.0.as_ref() {
            "e" | "_e" | "_each" => {
                let body: Rc<dyn Block> = Rc::clone(bb);
                Some((Rc::new(PdObj::Block(Rc::new(EachBlock { body }))), false))
            }
            "m" | "_m" | "_map" => {
                let body: Rc<dyn Block> = Rc::clone(bb);
                Some((Rc::new(PdObj::Block(Rc::new(MapBlock { body }))), false))
            }
            "u" | "_u" | "_under" => {
                let body: Rc<dyn Block> = Rc::clone(bb);
                Some((Rc::new(PdObj::Block(Rc::new(UnderBlock { body }))), false))
            }
            "k" | "_k" | "_keep" => {
                let body: Rc<dyn Block> = Rc::clone(bb);
                Some((Rc::new(PdObj::Block(Rc::new(KeepBlock { body }))), false))
            }
            "q" | "_q" | "_keepunder" => {
                let body: Rc<dyn Block> = Rc::clone(bb);
                Some((Rc::new(PdObj::Block(Rc::new(KeepUnderBlock { body }))), false))
            }
            _ => None
        }
        _ => None
    }
}

fn apply_all_trailers(mut obj: Rc<PdObj>, mut reluctant: bool, trailer: &[lex::Trailer]) -> Option<(Rc<PdObj>, bool)> {
    for t in trailer {
        let np = apply_trailer(&obj, t)?; // unwraps or returns None from entire function
        obj = np.0;
        reluctant = np.1;
    }
    Some((obj, reluctant))
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
    fn run(&self, mut env: &mut Environment) {
        for RcToken(leader, trailer) in &self.0 {
            // println!("{:?} {:?}", leader, trailer);
            // TODO: handle trailers lolololol
            let (obj, reluctant) = match leader {
                RcLeader::Lit(obj) => {
                    apply_all_trailers(Rc::clone(obj), true, trailer).expect("Could not apply trailers to literal")
                }
                RcLeader::Var(s) => {
                    let (obj, rest) = lookup_and_break_trailers(env, s, trailer).expect("Undefined variable!");
                    apply_all_trailers(Rc::clone(obj), false, rest).expect("Could not apply trailers to variable")
                }
            };

            if reluctant {
                env.push(obj);
            } else {
                apply_on(&mut env, obj);
            }
        }
    }
    fn code_repr(&self) -> String {
        "???".to_string()
    }
}

fn apply_on(env: &mut Environment, obj: Rc<PdObj>) {
    match &*obj {
        PdObj::Num(_)    => { env.push(obj); }
        PdObj::String(_) => { env.push(obj); }
        PdObj::List(_)   => { env.push(obj); }
        PdObj::Block(bb) => {
            bb.run(env);
        }
    }
}

macro_rules! n_n {
    ($a:ident, $x:expr) => {
        unary_num_case(|_, $a| vec![Rc::new(PdObj::Num($x))])
    };
}
macro_rules! nn_n {
    ($a:ident, $b:ident, $x:expr) => {
        binary_num_case(|_, $a, $b| vec![Rc::new(PdObj::Num($x))])
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

fn pd_deep_length(x: &PdObj) -> usize {
    match x {
        PdObj::Num(_) => 1,
        PdObj::String(ss) => ss.len(),
        PdObj::List(v) => v.iter().map(|x| pd_deep_length(&*x)).sum(),
        PdObj::Block(_) => { panic!("wtf not deep"); }
    }
}

fn pd_deep_sum(x: &PdObj) -> PdNum {
    match x {
        PdObj::Num(n) => n.clone(),
        PdObj::String(ss) => PdNum::Char(ss.iter().map(|x| BigInt::from(*x as u32)).sum()),
        PdObj::List(v) => v.iter().map(|x| pd_deep_sum(&*x)).sum(),
        PdObj::Block(_) => { panic!("wtf not deep"); }
    }
}

fn pd_deep_square_sum(x: &PdObj) -> PdNum {
    match x {
        PdObj::Num(n) => n * n,
        PdObj::String(ss) => PdNum::Char(ss.iter().map(|x| BigInt::from(*x as u32).pow(2u32)).sum()),
        PdObj::List(v) => v.iter().map(|x| pd_deep_square_sum(&*x)).sum(),
        PdObj::Block(_) => { panic!("wtf not deep"); }
    }
}

fn pd_deep_standard_deviation(x: &PdObj) -> Option<PdNum> {
    let c = PdNum::from(pd_deep_length(x));
    let s = pd_deep_sum(x);
    let q = pd_deep_square_sum(x);
    ((&(q - s.pow(2u32)) / &c) / (&c - &PdNum::from(1))).sqrt()
}

fn pd_deep_product(x: &PdObj) -> PdNum {
    match x {
        PdObj::Num(n) => n.clone(),
        PdObj::String(ss) => PdNum::Char(ss.iter().map(|x| BigInt::from(*x as u32)).product()),
        PdObj::List(v) => v.iter().map(|x| pd_deep_product(&*x)).product(),
        PdObj::Block(_) => { panic!("wtf not deep"); }
    }
}

fn bi_iverson(b: bool) -> BigInt { if b { BigInt::from(1) } else { BigInt::from(0) } }

fn initialize(env: &mut Environment) {
    let plus_case = nn_n![a, b, a + b];
    let minus_case = nn_n![a, b, a - b];
    let times_case = nn_n![a, b, a * b];
    // TODO: signs...
    let div_case = nn_n![a, b, a / b];
    let mod_case = nn_n![a, b, a % b];
    let intdiv_case = nn_n![a, b, a.div_floor(b)];

    let inc_case   = n_n![a, a.add_const( 1)];
    let dec_case   = n_n![a, a.add_const(-1)];
    let inc2_case  = n_n![a, a.add_const( 2)];
    let dec2_case  = n_n![a, a.add_const(-2)];

    let ceil_case  = n_n![a, a.ceil()];
    let floor_case = n_n![a, a.floor()];

    let lt_case = nn_n![a, b, PdNum::Int(bi_iverson(a < b))];
    let gt_case = nn_n![a, b, PdNum::Int(bi_iverson(a > b))];
    let min_case = nn_n![a, b, PdNum::clone(a.min(b))];
    let max_case = nn_n![a, b, PdNum::clone(a.max(b))];

    let uncons_case = unary_seq_range_case(|_, a| {
        let (x, xs) = a.split_first().expect("Uncons of empty list");
        vec![xs, x]
    });
    let first_case = unary_seq_range_case(|_, a| { vec![a.first().expect("First of empty list")] });
    let rest_case = unary_seq_range_case(|_, a| { vec![a.split_first().expect("Rest of empty list").1] });

    let unsnoc_case = unary_seq_range_case(|_, a| {
        let (x, xs) = a.split_last().expect("Unsnoc of empty list");
        vec![xs, x]
    });
    let last_case = unary_seq_range_case(|_, a| { vec![a.last().expect("Last of empty list")] });
    let butlast_case = unary_seq_range_case(|_, a| { vec![a.split_last().expect("Butlast of empty list").1] });

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
            vec![pd_list(pd_map(env, a, b.iter()))]
        }
    });

    let square_case   : Rc<dyn Case> = Rc::new(UnaryNumCase { func: |_, a| vec![Rc::new(PdObj::Num(a * a))] });

    macro_rules! cc {
        ($($case:expr),*) => {
            vec![$( Rc::clone(&$case), )*];
        }
    }

    add_cases("+", cc![plus_case]);
    add_cases("-", cc![minus_case]);
    add_cases("*", cc![times_case]);
    add_cases("/", cc![div_case]);
    add_cases("%", cc![mod_case, map_case]);
    add_cases("÷", cc![intdiv_case]);
    add_cases("(", cc![dec_case, uncons_case]);
    add_cases(")", cc![inc_case, unsnoc_case]);
    add_cases("<", cc![lt_case]);
    add_cases(">", cc![gt_case]);
    add_cases("<m", cc![min_case]);
    add_cases(">m", cc![max_case]);
    add_cases("Õ", cc![min_case]);
    add_cases("Ã", cc![max_case]);
    add_cases("‹", cc![floor_case, first_case]);
    add_cases("›", cc![ceil_case, last_case]);
    add_cases("«", cc![dec2_case, butlast_case]);
    add_cases("»", cc![inc2_case, rest_case]);
    add_cases("²", cc![square_case]);

    let dup_case   : Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| cc![a, a] });
    add_cases(":", cc![dup_case]);
    let dup_pair_case: Rc<dyn Case> = Rc::new(BinaryAnyCase { func: |_, a, b| cc![a, b, a, b] });
    add_cases(":p", cc![dup_pair_case]);
    let dup_around_case: Rc<dyn Case> = Rc::new(BinaryAnyCase { func: |_, a, b| cc![a, b, a] });
    add_cases(":a", cc![dup_around_case]);
    let swap_case: Rc<dyn Case> = Rc::new(BinaryAnyCase { func: |_, a, b| cc![b, a] });
    add_cases("\\", cc![swap_case]);
    let swap_in_case: Rc<dyn Case> = Rc::new(TernaryAnyCase { func: |_, a, b, c| cc![c, a, b] });
    add_cases("\\i", cc![swap_in_case]);
    let swap_out_case: Rc<dyn Case> = Rc::new(TernaryAnyCase { func: |_, a, b, c| cc![b, c, a] });
    add_cases("\\o", cc![swap_out_case]);

    let pop_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, _a| cc![] });
    add_cases(";", cc![pop_case]);
    let pop_out_case: Rc<dyn Case> = Rc::new(TernaryAnyCase { func: |_, _a, b, c| cc![b, c] });
    add_cases(";o", cc![pop_out_case]);
    let pop_pair_case: Rc<dyn Case> = Rc::new(TernaryAnyCase { func: |_, _a, _b, c| cc![c] });
    add_cases(";p", cc![pop_pair_case]);
    let pop_around_case: Rc<dyn Case> = Rc::new(TernaryAnyCase { func: |_, _a, b, _c| cc![b] });
    add_cases(";a", cc![pop_around_case]);
    let pop_under_case: Rc<dyn Case> = Rc::new(BinaryAnyCase { func: |_, _a, b| cc![b] });
    add_cases("¸", cc![pop_under_case]);

    let pack_one_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| cc![pd_list(vec![Rc::clone(a)])] });
    add_cases("†", cc![pack_one_case]);
    let pack_two_case: Rc<dyn Case> = Rc::new(BinaryAnyCase { func: |_, a, b| cc![pd_list(vec![Rc::clone(a), Rc::clone(b)])] });
    add_cases("‡", cc![pack_two_case]);
    let not_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| cc![Rc::new(PdObj::from(bi_iverson(!pd_truthy(a))))] });
    add_cases("!", cc![not_case]);

    let sum_case   : Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| cc![Rc::new(PdObj::Num(pd_deep_sum(a)))] });
    add_cases("Š", cc![sum_case]);

    let product_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| cc![Rc::new(PdObj::Num(pd_deep_product(a)))] });
    add_cases("Þ", cc![product_case]);

    let average_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| cc![Rc::new(PdObj::Num(pd_deep_sum(a) / PdNum::Float(pd_deep_length(a) as f64)))] });
    add_cases("Av", cc![average_case]);

    let hypotenuse_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| cc![Rc::new(PdObj::Num(pd_deep_square_sum(a).sqrt().expect("sqrt in hypotenuse failed")))] });
    add_cases("Hy", cc![hypotenuse_case]);

    let standard_deviation_case: Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| cc![Rc::new(PdObj::Num(pd_deep_standard_deviation(a).expect("sqrt in hypotenuse failed")))] });
    add_cases("Sg", cc![standard_deviation_case]);

    // env.variables.insert("X".to_string(), Rc::new(PdObj::Int(3.to_bigint().unwrap())));
    env.short_insert("N", PdObj::from('\n'));
    env.short_insert("A", PdObj::from(10));
    env.short_insert("¹", PdObj::from(11));
    env.short_insert("∅", PdObj::from(0));
    env.short_insert("α", PdObj::from(1));
    env.short_insert("Ep", PdObj::Num(PdNum::Float(1e-9)));

    env.short_insert(" ", PdObj::Block(Rc::new(BuiltIn {
        name: "Nop".to_string(),
        func: |_env| {},
    })));
    env.short_insert("[", PdObj::Block(Rc::new(BuiltIn {
        name: "Mark_stack".to_string(),
        func: |env| { env.mark_stack(); },
    })));
    env.short_insert("]", PdObj::Block(Rc::new(BuiltIn {
        name: "Pack".to_string(),
        func: |env| {
            let list = env.pop_until_stack_marker();
            env.push(Rc::new(PdObj::List(Rc::new(list))));
        },
    })));
    env.short_insert("¬", PdObj::Block(Rc::new(BuiltIn {
        name: "Pack_reverse".to_string(),
        func: |env| {
            let mut list = env.pop_until_stack_marker();
            list.reverse();
            env.push(Rc::new(PdObj::List(Rc::new(list))));
        },
    })));
    env.short_insert("~", PdObj::Block(Rc::new(BuiltIn {
        name: "Expand_or_eval".to_string(),
        func: |env| {
            match &*env.pop_or_panic("~ failed") {
                PdObj::Block(bb) => { bb.run(env); }
                PdObj::List(ls) => { env.extend_clone(ls); }
                _ => { panic!("~ can't handle"); }
            }
        },
    })));
    env.short_insert("O", PdObj::Block(Rc::new(BuiltIn {
        name: "Print".to_string(),
        func: |env| {
            let obj = env.pop_or_panic("O failed");
            print!("{}", env.to_string(&obj));
        },
    })));
    env.short_insert("P", PdObj::Block(Rc::new(BuiltIn {
        name: "Print".to_string(),
        func: |env| {
            let obj = env.pop_or_panic("P failed");
            println!("{}", env.to_string(&obj));
        },
    })));
}

pub fn simple_eval(code: &str) -> Vec<Rc<PdObj>> {
    let mut env = Environment::new();
    initialize(&mut env);

    let block = CodeBlock(rcify(lex::parse(code)));

    block.run(&mut env);

    env.stack
}
