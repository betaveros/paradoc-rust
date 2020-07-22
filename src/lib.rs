#[macro_use] extern crate lazy_static;

use std::rc::Rc;
use std::fmt::Debug;
use std::mem;
use num::Integer;
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

#[derive(Debug)]
pub struct ShadowState {
    env: Box<Environment>,
    arity: usize,
}

impl ShadowState {
    fn pop(&mut self) -> Option<Rc<PdObj>> {
        let res = self.env.pop();
        if res.is_some() { self.arity += 1; }
        res
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

    fn short_insert(&mut self, name: &str, obj: PdObj) {
        self.variables.insert(name.to_string(), Rc::new(obj));
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

    fn run_on_bracketed_shadow<T>(&mut self, body: impl FnOnce(&mut Environment) -> T) -> T {
        let env = mem::replace(self, Environment::new());

        let mut benv = Environment {
            stack: Vec::new(),
            x_stack: Vec::new(),
            variables: HashMap::new(),
            marker_stack: vec![0],
            shadow: Some(ShadowState { env: Box::new(env), arity: 0 }),
        };

        let ret = body(&mut benv);

        mem::replace(self, *(benv.shadow.unwrap().env));

        ret
    }
}

pub trait Block : Debug {
    fn run(&self, env: &mut Environment) -> ();
    fn code_repr(&self) -> String;
}

fn sandbox(env: &mut Environment, func: &Rc<dyn Block>, args: Vec<Rc<PdObj>>) -> Vec<Rc<PdObj>> {
    env.run_on_bracketed_shadow(|inner| {
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
    PdList(Vec<Rc<PdObj>>),
    PdBlock(Rc<dyn Block>),
}

impl PartialEq for PdObj {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PdObj::PdInt   (a), PdObj::PdInt   (b)) => a == b,
            (PdObj::PdFloat (a), PdObj::PdFloat (b)) => a == b,
            (PdObj::PdChar  (a), PdObj::PdChar  (b)) => a == b,
            (PdObj::PdString(a), PdObj::PdString(b)) => a == b,
            (PdObj::PdList  (a), PdObj::PdList  (b)) => a == b,
            _ => false,
        }
    }
}

impl From<i32> for PdObj {
    fn from(x: i32) -> Self { PdObj::PdInt(x.to_bigint().unwrap()) }
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
        PdObj::PdInt(a) => Some(a.to_f64().unwrap()), // FIXME
        // (we do not want to propagate the option since cases would confusingly fail to apply)
        PdObj::PdFloat(a) => Some(*a),
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
fn binary_int_case(func: fn(&mut Environment, &BigInt, &BigInt) -> Vec<Rc<PdObj>>) -> Rc<dyn Case> {
    Rc::new(BinaryCase { coerce1: just_int, coerce2: just_int, func })
}
fn binary_floatify_case(func: fn(&mut Environment, &f64, &f64) -> Vec<Rc<PdObj>>) -> Rc<dyn Case> {
    Rc::new(BinaryCase { coerce1: floatify, coerce2: floatify, func })
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
                None => {}
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
        match env.pop() {
            None => {
                panic!("each no stack")
            }
            Some(top) => match &*top {
                PdObj::PdList(vec) => {
                    for obj in vec {
                        env.push(Rc::clone(obj));
                        self.body.run(env);
                    }
                }
                _ => { panic!("each failed") }
            }
        }
    }
    fn code_repr(&self) -> String {
        self.body.code_repr() + "_each"
    }
}

#[derive(Debug)]
struct MapBlock {
    body: Rc<dyn Block>,
}
impl Block for MapBlock {
    fn run(&self, env: &mut Environment) {
        // TODO: literally everything
        // Extract into sandbox; push x-stack; handle continue/breaks
        match env.pop() {
            None => panic!("map no stack"),
            Some(top) => match &*top {
                PdObj::PdList(vec) => {
                    let res = vec.iter().flat_map(|obj| {
                        sandbox(env, &self.body, vec![Rc::clone(obj)])
                    }).collect();
                    env.push(Rc::new(PdObj::PdList(res)));
                }
                _ => { panic!("each failed") }
            }
        }
    }
    fn code_repr(&self) -> String {
        self.body.code_repr() + "_each"
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

    if let Some(res) = env.variables.get(leader) {
        last_found = Some((res, trailers)); // lowest priority
    }

    for (i, t) in trailers.iter().enumerate() {
        var.push_str(&t.0);

        if let Some(res) = env.variables.get(&var) {
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
                    apply_all_trailers(Rc::clone(obj), true, trailer).unwrap()
                }
                RcLeader::Var(s) => {
                    match lookup_and_break_trailers(env, s, trailer) {
                        Some((obj, rest)) => {
                            println!("test {:?} {:?}", obj, rest);
                            apply_all_trailers(Rc::clone(obj), false, rest).unwrap()
                        }
                        None => { panic!("undefined var"); }
                    }
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

macro_rules! ii_i {
    ($a:ident, $b:ident, $x:expr) => {
        binary_int_case(|_, $a, $b| vec![Rc::new(PdObj::PdInt($x))])
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

    let inc_case   : Rc<dyn Case> = Rc::new(UnaryIntCase { func: |_, a| vec![Rc::new(PdObj::PdInt(a + 1))] });
    let dec_case   : Rc<dyn Case> = Rc::new(UnaryIntCase { func: |_, a| vec![Rc::new(PdObj::PdInt(a - 1))] });

    let mut add_cases = |name: &str, cases: Vec<Rc<dyn Case>>| {
        env.variables.insert(name.to_string(), Rc::new(PdObj::PdBlock(Rc::new(CasedBuiltIn {
            name: name.to_string(),
            cases,
        }))));
    };

    macro_rules! cc {
        ($($case:expr),*) => {
            vec![$( Rc::clone(&$case), )*];
        }
    }

    add_cases("+", cc![plus_case, fplus_case]);
    add_cases("-", cc![minus_case, fminus_case]);
    add_cases("*", cc![times_case, ftimes_case]);
    add_cases("/", cc![div_case]);
    add_cases("%", cc![mod_case, fmod_case]);
    add_cases("÷", cc![intdiv_case, fintdiv_case]);
    add_cases("(", cc![dec_case]);
    add_cases(")", cc![inc_case]);

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

    // env.variables.insert("X".to_string(), Rc::new(PdObj::PdInt(3.to_bigint().unwrap())));
    env.short_insert("N", PdObj::PdChar('\n'));
    env.short_insert("A", PdObj::from(10));
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
            env.push(Rc::new(PdObj::PdList(list)));
        },
    })));
    env.short_insert("~", PdObj::PdBlock(Rc::new(BuiltIn {
        name: "Expand_or_eval".to_string(),
        func: |env| {
            match env.pop() {
                None => { panic!("~_~"); }
                Some(x) => {
                    match &*x {
                        PdObj::PdBlock(bb) => {
                            bb.run(env);
                        }
                        PdObj::PdList(ls) => {
                            env.extend_clone(ls);
                        }
                        _ => {
                            panic!("~ what");
                        }
                    }
                }
            }
        },
    })));
    env.short_insert("O", PdObj::PdBlock(Rc::new(BuiltIn {
        name: "Print".to_string(),
        func: |env| {
            match env.pop() {
                None => { panic!("~_~"); }
                Some(x) => {
                    print!("{}", env.to_string(&x));
                }
            }
        },
    })));
    env.short_insert("P", PdObj::PdBlock(Rc::new(BuiltIn {
        name: "Print".to_string(),
        func: |env| {
            match env.pop() {
                None => { panic!("~_~"); }
                Some(x) => {
                    println!("{}", env.to_string(&x));
                }
            }
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
