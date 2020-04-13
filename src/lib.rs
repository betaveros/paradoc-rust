use std::rc::Rc;
use std::fmt::Debug;
use num::bigint::BigInt;
use num::bigint::ToBigInt;
use num_traits::cast::ToPrimitive;
use regex::Regex;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Environment {
    stack: Vec<Rc<PdObj>>,
    x_stack: Vec<Rc<PdObj>>,
    variables: HashMap<String, Rc<PdObj>>,
    marker_stack: Vec<usize>,
}

impl Environment {
    fn push(&mut self, obj: Rc<PdObj>) {
        self.stack.push(obj)
    }
    fn extend(&mut self, objs: Vec<Rc<PdObj>>) {
        for obj in objs {
            self.push(obj)
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
            None => None
        }
        // TODO: stack trigger
    }

    fn new() -> Environment {
        Environment {
            stack: Vec::new(),
            x_stack: Vec::new(),
            variables: HashMap::new(),
            marker_stack: Vec::new(),
        }
    }
}

pub trait Block : Debug {
    fn run(&self, env: &mut Environment) -> ();
    fn code_repr(&self) -> &String;
}

#[derive(Debug)]
pub enum PdObj {
    PdInt(BigInt),
    PdFloat(f64),
    PdChar(char),
    PdString(String),
    PdList(Vec<Rc<PdObj>>),
    PdBlock(Box<dyn Block>),
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
    fn code_repr(&self) -> &String {
        &self.name
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
    fn code_repr(&self) -> &String {
        &self.name
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
    let (div_case,   fdiv_case  ) = numfs![a, b, a / b];
    let inc_case   : Rc<dyn Case> = Rc::new(UnaryIntCase { func: |_, a| vec![Rc::new(PdObj::PdInt(a + 1))] });
    let dec_case   : Rc<dyn Case> = Rc::new(UnaryIntCase { func: |_, a| vec![Rc::new(PdObj::PdInt(a - 1))] });
    let dup_case   : Rc<dyn Case> = Rc::new(UnaryAnyCase { func: |_, a| vec![Rc::clone(a), Rc::clone(a)] });

    let mut add_cases = |name: &str, cases: Vec<Rc<dyn Case>>| {
        env.variables.insert(name.to_string(), Rc::new(PdObj::PdBlock(Box::new(CasedBuiltIn {
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
    add_cases("/", cc![div_case, fdiv_case]);
    add_cases("(", cc![inc_case]);
    add_cases(")", cc![dec_case]);
    add_cases(":", cc![dup_case]);

    // env.variables.insert("X".to_string(), Rc::new(PdObj::PdInt(3.to_bigint().unwrap())));
    env.variables.insert("N".to_string(), Rc::new(PdObj::PdChar('\n')));
    env.variables.insert("A".to_string(), Rc::new(PdObj::PdInt(10.to_bigint().unwrap())));
    env.variables.insert("Ep".to_string(), Rc::new(PdObj::PdFloat(1e-9)));
}

pub fn simple_eval(code: &str) -> Vec<Rc<PdObj>> {
    let mut env = Environment::new();
    initialize(&mut env);

    let pd_token_pattern = Regex::new(r#"\.\.[^\n\r]*|("(?:\\"|\\\\|[^"])*"|'.|[0-9]+(?:\.[0-9]+)?(?:e[0-9]+)?|[^"'0-9a-z_])([a-z_]*)"#).unwrap();
    let mut i = 0;
    let mut go = true;
    while go {
        match pd_token_pattern.find_at(code, i) {
            Some(match_result) => {
                let j = match_result.end();
                let token = &code[i..j];
                println!("{}", token);
                match token.parse::<BigInt>() {
                    Ok(x) => {
                        env.push(Rc::new(PdObj::PdInt(x.clone())));
                    }
                    _ => {}
                }
                match env.variables.get(token) {
                    Some(obj) => {
                        let obj2 = Rc::clone(obj);
                        apply_on(&mut env, obj2);
                    }
                    None => {}
                }
                // println!("{:?}", &code[i..j].parse::<BigInt>());
                i = j;
            },
            None => {
                go = false;
            },
        }
    }

    env.stack
}
