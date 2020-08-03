#[macro_use] extern crate lazy_static;

use std::cmp::Ordering;
use std::rc::Rc;
use std::slice::Iter;
use std::fmt::Debug;
use std::mem;
use num::Integer;
use num_iter;
use num::bigint::BigInt;
use num::bigint::ToBigInt;
use num_traits::cast::ToPrimitive;
use std::collections::HashMap;

mod lex;

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
        self.push_x(Rc::new(PdObj::PdString("INTERNAL Y FILLER -- YOU SHOULD NOT SEE THIS".to_string())));
        self.push_x(Rc::new(PdObj::PdString("INTERNAL X FILLER -- YOU SHOULD NOT SEE THIS".to_string())));
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
            PdObj::PdInt(a) => a.to_string(),
            PdObj::PdFloat(x) => x.to_string(),
            PdObj::PdChar(c) => c.to_string(),
            PdObj::PdString(s) => s.clone(),
            PdObj::PdList(v) => v.iter().map(|o| self.to_string(o)).collect::<Vec<String>>().join(""),
            PdObj::PdBlock(b) => b.code_repr(),
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
    PdInt(BigInt),
    PdFloat(f64),
    PdChar(char),
    PdString(String),
    PdList(Rc<Vec<Rc<PdObj>>>),
    PdBlock(Rc<dyn Block>),
}

// this seems... nontrivial??
fn cmp_bigint_f64(a: &BigInt, b: &f64) -> Option<Ordering> {
    if let Some(bi) = b.to_bigint() {
        Some(a.cmp(&bi))
    } else {
        b.floor().to_bigint().map(|bi| {
            match a.cmp(&bi) {
                Ordering::Less    => Ordering::Less,
                Ordering::Equal   => Ordering::Less,
                Ordering::Greater => Ordering::Greater,
            }
        })
    }
}

impl PartialEq for PdObj {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PdObj::PdInt   (a), PdObj::PdInt   (b)) => a == b,
            (PdObj::PdInt   (a), PdObj::PdFloat (b)) => b.to_bigint().map_or(false, |x| &x == a),
            (PdObj::PdInt   (a), PdObj::PdChar  (b)) => a == &BigInt::from(*b as u32),
            (PdObj::PdFloat (a), PdObj::PdInt   (b)) => a.to_bigint().map_or(false, |x| &x == b),
            (PdObj::PdFloat (a), PdObj::PdFloat (b)) => a == b,
            (PdObj::PdFloat (a), PdObj::PdChar  (b)) => a == &(*b as u32 as f64),
            (PdObj::PdChar  (a), PdObj::PdInt   (b)) => &BigInt::from(*a as u32) == b,
            (PdObj::PdChar  (a), PdObj::PdFloat (b)) => &(*a as u32 as f64) == b,
            (PdObj::PdChar  (a), PdObj::PdChar  (b)) => a == b,
            (PdObj::PdString(a), PdObj::PdString(b)) => a == b,
            (PdObj::PdList  (a), PdObj::PdList  (b)) => a == b,
            _ => false,
        }
    }
}

// TODO: Watch https://github.com/rust-lang/rust/issues/72599, we will probably want total
// orderings in some cases.

impl PartialOrd for PdObj {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (PdObj::PdInt   (a), PdObj::PdInt   (b)) => Some(a.cmp(b)),
            (PdObj::PdInt   (a), PdObj::PdFloat (b)) => cmp_bigint_f64(a, b),
            (PdObj::PdInt   (a), PdObj::PdChar  (b)) => Some(a.cmp(&BigInt::from(*b as u32))),
            (PdObj::PdFloat (a), PdObj::PdInt   (b)) => cmp_bigint_f64(b, a).map(|ord| ord.reverse()),
            (PdObj::PdFloat (a), PdObj::PdFloat (b)) => a.partial_cmp(b),
            (PdObj::PdFloat (a), PdObj::PdChar  (b)) => a.partial_cmp(&(*b as u32 as f64)), // :thonk:
            (PdObj::PdChar  (a), PdObj::PdInt   (b)) => Some(BigInt::from(*a as u32).cmp(b)),
            (PdObj::PdChar  (a), PdObj::PdFloat (b)) => (*a as u32 as f64).partial_cmp(b), // :thonk:
            (PdObj::PdChar  (a), PdObj::PdChar  (b)) => Some(a.cmp(b)),
            (PdObj::PdString(a), PdObj::PdString(b)) => Some(a.cmp(b)),
            (PdObj::PdList  (a), PdObj::PdList  (b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

impl From<i32> for PdObj {
    fn from(x: i32) -> Self { PdObj::PdInt(BigInt::from(x)) }
}
impl From<usize> for PdObj {
    fn from(x: usize) -> Self { PdObj::PdInt(BigInt::from(x)) }
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

fn just_int(obj: &PdObj) -> Option<BigInt> {
    match obj {
        PdObj::PdInt(a) => Some(a.clone()),
        _ => None,
    }
}
fn floatify(obj: &PdObj) -> Option<f64> {
    match obj {
        PdObj::PdInt(a) => Some(a.to_f64().expect("BigInt too large to floatify?")), // FIXME
        // (we do not want to propagate the option since cases would confusingly fail to apply)
        PdObj::PdFloat(a) => Some(*a),
        _ => None,
    }
}
fn seq_range(obj: &PdObj) -> Option<Rc<Vec<Rc<PdObj>>>> {
    match obj {
        // TODO: wasteful to construct the vector :(
        PdObj::PdInt(a) => Some(Rc::new(num_iter::range(BigInt::from(0), a.clone()).map(|x| Rc::new(PdObj::PdInt(x))).collect())),
        // PdObj::PdInt(a) => Some((BigInt::from(0)..a.clone()).collect()),
        PdObj::PdList(a) => Some(Rc::clone(a)),
        _ => None,
    }
}
fn just_block(obj: &PdObj) -> Option<Rc<dyn Block>> {
    match obj {
        PdObj::PdBlock(a) => Some(Rc::clone(a)),
        _ => None,
    }
}

struct UnaryIntCase {
    func: fn(&mut Environment, &BigInt) -> Vec<Rc<PdObj>>,
}
impl Case for UnaryIntCase {
    fn arity(&self) -> usize { 1 }
    fn maybe_run_noncommutatively(&self, env: &mut Environment, args: &Vec<Rc<PdObj>>) -> Option<Vec<Rc<PdObj>>> {
        match &*args[0] {
            PdObj::PdInt(ai) => {
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
fn unary_int_case(func: fn(&mut Environment, &BigInt) -> Vec<Rc<PdObj>>) -> Rc<dyn Case> {
    Rc::new(UnaryCase { coerce: just_int, func })
}
fn unary_floatify_case(func: fn(&mut Environment, &f64) -> Vec<Rc<PdObj>>) -> Rc<dyn Case> {
    Rc::new(UnaryCase { coerce: floatify, func })
}
fn binary_int_case(func: fn(&mut Environment, &BigInt, &BigInt) -> Vec<Rc<PdObj>>) -> Rc<dyn Case> {
    Rc::new(BinaryCase { coerce1: just_int, coerce2: just_int, func })
}
fn binary_floatify_case(func: fn(&mut Environment, &f64, &f64) -> Vec<Rc<PdObj>>) -> Rc<dyn Case> {
    Rc::new(BinaryCase { coerce1: floatify, coerce2: floatify, func })
}
fn unary_seq_range_case(func: fn(&mut Environment, &Rc<Vec<Rc<PdObj>>>) -> Vec<Rc<PdObj>>) -> Rc<dyn Case> {
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
                for obj in &**vec {
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
fn pd_map(env: &mut Environment, func: &Rc<dyn Block>, it: Iter<Rc<PdObj>>) -> Vec<Rc<PdObj>> {
    env.push_yx();
    let res = it.enumerate().flat_map(|(i, obj)| {
        env.set_yx(Rc::new(PdObj::from(i)), Rc::clone(obj));
        sandbox(env, &func, vec![Rc::clone(obj)])
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
                env.push(Rc::new(PdObj::PdList(Rc::new(res))));
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
                RcLeader::Lit(Rc::new(PdObj::PdString(s)))
            }
            lex::Leader::IntLit(n) => {
                RcLeader::Lit(Rc::new(PdObj::PdInt(n)))
            }
            lex::Leader::CharLit(c) => {
                RcLeader::Lit(Rc::new(PdObj::PdChar(c)))
            }
            lex::Leader::FloatLit(f) => {
                RcLeader::Lit(Rc::new(PdObj::PdFloat(f)))
            }
            lex::Leader::Block(b) => {
                RcLeader::Lit(Rc::new(PdObj::PdBlock(Rc::new(CodeBlock(rcify(*b))))))
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
        PdObj::PdBlock(bb) => match trailer.0.as_ref() {
            "e" | "_e" | "_each" => {
                let body: Rc<dyn Block> = Rc::clone(bb);
                Some((Rc::new(PdObj::PdBlock(Rc::new(EachBlock { body }))), false))
            }
            "m" | "_m" | "_map" => {
                let body: Rc<dyn Block> = Rc::clone(bb);
                Some((Rc::new(PdObj::PdBlock(Rc::new(MapBlock { body }))), false))
            }
            "u" | "_u" | "_under" => {
                let body: Rc<dyn Block> = Rc::clone(bb);
                Some((Rc::new(PdObj::PdBlock(Rc::new(UnderBlock { body }))), false))
            }
            "k" | "_k" | "_keep" => {
                let body: Rc<dyn Block> = Rc::clone(bb);
                Some((Rc::new(PdObj::PdBlock(Rc::new(KeepBlock { body }))), false))
            }
            "q" | "_q" | "_keepunder" => {
                let body: Rc<dyn Block> = Rc::clone(bb);
                Some((Rc::new(PdObj::PdBlock(Rc::new(KeepUnderBlock { body }))), false))
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
        PdObj::PdInt(_)    => { env.push(obj); }
        PdObj::PdString(_) => { env.push(obj); }
        PdObj::PdChar(_)   => { env.push(obj); }
        PdObj::PdList(_)   => { env.push(obj); }
        PdObj::PdFloat(_)  => { env.push(obj); }
        PdObj::PdBlock(bb) => {
            bb.run(env);
        }
    }
}

macro_rules! i_i {
    ($a:ident, $x:expr) => {
        unary_int_case(|_, $a| vec![Rc::new(PdObj::PdInt($x))])
    };
}
macro_rules! f_f {
    ($a:ident, $x:expr) => {
        unary_floatify_case(|_, $a| vec![Rc::new(PdObj::PdFloat($x))])
    };
}
macro_rules! f_i {
    ($a:ident, $x:expr) => {
        unary_floatify_case(|_, $a| vec![Rc::new(PdObj::PdInt($x))])
    };
}
macro_rules! ii_i {
    ($a:ident, $b:ident, $x:expr) => {
        binary_int_case(|_, $a, $b| vec![Rc::new(PdObj::PdInt($x))])
    };
}
macro_rules! ff_i {
    ($a:ident, $b:ident, $x:expr) => {
        binary_floatify_case(|_, $a, $b| vec![Rc::new(PdObj::PdInt($x))])
    };
}
macro_rules! ff_f {
    ($a:ident, $b:ident, $x:expr) => {
        binary_floatify_case(|_, $a, $b| vec![Rc::new(PdObj::PdFloat($x))])
    };
}
macro_rules! numfs {
    ($a:ident, $b:ident, $x:expr) => {
        (ii_i![$a, $b, $x], ff_f![$a, $b, $x])
    }
}

fn pd_list(xs: Vec<Rc<PdObj>>) -> Rc<PdObj> { Rc::new(PdObj::PdList(Rc::new(xs))) }

fn bi_iverson(b: bool) -> BigInt { if b { BigInt::from(1) } else { BigInt::from(0) } }

fn initialize(env: &mut Environment) {
    let (plus_case,  fplus_case ) = numfs![a, b, a + b];
    let (minus_case, fminus_case) = numfs![a, b, a - b];
    let (times_case, ftimes_case) = numfs![a, b, a * b];
    // TODO: signs...
    let div_case = ff_f![a, b, a / b];
    let mod_case = ii_i![a, b, Integer::mod_floor(a, b)];
    let fmod_case = ff_f![a, b, a.rem_euclid(*b)];
    let intdiv_case = ii_i![a, b, Integer::div_floor(a, b)];
    let fintdiv_case = ff_f![a, b, a.div_euclid(*b)];

    let id_int_case = i_i![a, a.clone()];
    let inc_case   = i_i![a, a + 1];
    let dec_case   = i_i![a, a - 1];
    let inc2_case  = i_i![a, a + 2];
    let dec2_case  = i_i![a, a - 2];
    let finc_case  = f_f![a, a + 1.0];
    let fdec_case  = f_f![a, a - 1.0];
    let finc2_case = f_f![a, a + 2.0];
    let fdec2_case = f_f![a, a - 2.0];

    let fceil_case  = f_i![a, a.ceil().to_bigint().expect("Ceiling of float was not integer")];
    let ffloor_case = f_i![a, a.floor().to_bigint().expect("Floor of float was not integer")];

    let ilt_case = ii_i![a, b, bi_iverson(a < b)];
    let flt_case = ff_i![a, b, bi_iverson(a < b)];
    let igt_case = ii_i![a, b, bi_iverson(a > b)];
    let fgt_case = ff_i![a, b, bi_iverson(a > b)];

    let uncons_case = unary_seq_range_case(|_, a| {
        let (x, xs) = a.split_first().expect("Uncons of empty list");
        vec![pd_list(xs.to_vec()), Rc::clone(x)]
    });
    let first_case = unary_seq_range_case(|_, a| { vec![Rc::clone(a.first().expect("First of empty list"))] });
    let rest_case = unary_seq_range_case(|_, a| { vec![pd_list(a.split_first().expect("Rest of empty list").1.to_vec())] });

    let unsnoc_case = unary_seq_range_case(|_, a| {
        let (x, xs) = a.split_last().expect("Unsnoc of empty list");
        vec![pd_list(xs.to_vec()), Rc::clone(x)]
    });
    let last_case = unary_seq_range_case(|_, a| { vec![Rc::clone(a.last().expect("Last of empty list"))] });
    let butlast_case = unary_seq_range_case(|_, a| { vec![pd_list(a.split_last().expect("Butlast of empty list").1.to_vec())] });

    let mut add_cases = |name: &str, cases: Vec<Rc<dyn Case>>| {
        env.variables.insert(name.to_string(), Rc::new(PdObj::PdBlock(Rc::new(CasedBuiltIn {
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

    let square_case   : Rc<dyn Case> = Rc::new(UnaryIntCase { func: |_, a| vec![Rc::new(PdObj::PdInt(a * a))] });

    macro_rules! cc {
        ($($case:expr),*) => {
            vec![$( Rc::clone(&$case), )*];
        }
    }

    add_cases("+", cc![plus_case, fplus_case]);
    add_cases("-", cc![minus_case, fminus_case]);
    add_cases("*", cc![times_case, ftimes_case]);
    add_cases("/", cc![div_case]);
    add_cases("%", cc![mod_case, fmod_case, map_case]);
    add_cases("÷", cc![intdiv_case, fintdiv_case]);
    add_cases("(", cc![dec_case, fdec_case, uncons_case]);
    add_cases(")", cc![inc_case, finc_case, unsnoc_case]);
    add_cases("<", cc![ilt_case, flt_case]);
    add_cases(">", cc![igt_case, fgt_case]);
    add_cases("‹", cc![id_int_case, ffloor_case, first_case]);
    add_cases("‹", cc![id_int_case, ffloor_case, first_case]);
    add_cases("›", cc![id_int_case, fceil_case, last_case]);
    add_cases("«", cc![dec2_case, fdec2_case, butlast_case]);
    add_cases("»", cc![inc2_case, finc2_case, rest_case]);
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

    // env.variables.insert("X".to_string(), Rc::new(PdObj::PdInt(3.to_bigint().unwrap())));
    env.short_insert("N", PdObj::PdChar('\n'));
    env.short_insert("A", PdObj::from(10));
    env.short_insert("¹", PdObj::from(11));
    env.short_insert("∅", PdObj::from(0));
    env.short_insert("α", PdObj::from(1));
    env.short_insert("Ep", PdObj::PdFloat(1e-9));

    env.short_insert(" ", PdObj::PdBlock(Rc::new(BuiltIn {
        name: "Nop".to_string(),
        func: |_env| {},
    })));
    env.short_insert("[", PdObj::PdBlock(Rc::new(BuiltIn {
        name: "Mark_stack".to_string(),
        func: |env| { env.mark_stack(); },
    })));
    env.short_insert("]", PdObj::PdBlock(Rc::new(BuiltIn {
        name: "Make_array".to_string(),
        func: |env| {
            let list = env.pop_until_stack_marker();
            env.push(Rc::new(PdObj::PdList(Rc::new(list))));
        },
    })));
    env.short_insert("~", PdObj::PdBlock(Rc::new(BuiltIn {
        name: "Expand_or_eval".to_string(),
        func: |env| {
            match &*env.pop_or_panic("~ failed") {
                PdObj::PdBlock(bb) => { bb.run(env); }
                PdObj::PdList(ls) => { env.extend_clone(ls); }
                _ => { panic!("~ can't handle"); }
            }
        },
    })));
    env.short_insert("O", PdObj::PdBlock(Rc::new(BuiltIn {
        name: "Print".to_string(),
        func: |env| {
            let obj = env.pop_or_panic("O failed");
            print!("{}", env.to_string(&obj));
        },
    })));
    env.short_insert("P", PdObj::PdBlock(Rc::new(BuiltIn {
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
