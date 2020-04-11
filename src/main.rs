use std::rc::Rc;
use std::fmt::Debug;
use num::bigint::BigInt;
use num::bigint::ToBigInt;
use regex::Regex;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Environment {
    stack: Vec<Rc<PdObj>>,
    x_stack: Vec<Rc<PdObj>>,
    variables: HashMap<String, Rc<PdObj>>,
}

pub trait Block : Debug {
    fn run(&self, env: &mut Environment) -> ();
    fn code_repr(&self) -> &String;
}

#[derive(Debug)]
enum PdObj {
    PdInt(BigInt),
    PdChar(char),
    PdString(String),
    PdList(Vec<Rc<PdObj>>),
    PdBlock(Box<dyn Block>),
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

trait Case {
    fn arity(&self) -> usize;
    fn maybe_run_noncommutatively(&self, env: &mut Environment) -> bool;
}

struct UnaryIntCase {
    func: fn(&mut Environment, &BigInt) -> Vec<Rc<PdObj>>,
}
impl Case for UnaryIntCase {
    fn arity(&self) -> usize { 1 }
    fn maybe_run_noncommutatively(&self, env: &mut Environment) -> bool {
        let len = env.stack.len();
        if len >= 1 {
            match &*env.stack[len - 1] {
                PdObj::PdInt(ai) => {
                    let ai2 = ai.clone();
                    let xs = (self.func)(env, &ai2);
                    env.stack.pop();
                    env.stack.extend(xs);
                    true
                }
                _ => false
            }
        } else {
            false
        }
    }
}

struct BinaryIntCase {
    func: fn(&mut Environment, &BigInt, &BigInt) -> Vec<Rc<PdObj>>,
}
impl Case for BinaryIntCase {
    fn arity(&self) -> usize { 2 }
    fn maybe_run_noncommutatively(&self, env: &mut Environment) -> bool {
        let len = env.stack.len();
        if len >= 2 {
            match (&*env.stack[len - 2], &*env.stack[len - 1]) {
                (PdObj::PdInt(ai), PdObj::PdInt(bi)) => {
                    let ai2 = ai.clone();
                    let bi2 = bi.clone();
                    let xs = (self.func)(env, &ai2, &bi2);
                    env.stack.pop();
                    env.stack.pop();
                    env.stack.extend(xs);
                    true
                }
                _ => false
            }
        } else {
            false
        }
    }
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
        for case in &self.cases {
            if case.maybe_run_noncommutatively(env) {
                done = true; break
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
        PdObj::PdInt(_)    => { env.stack.push(obj); }
        PdObj::PdString(_) => { env.stack.push(obj); }
        PdObj::PdChar(_)   => { env.stack.push(obj); }
        PdObj::PdList(_)   => { env.stack.push(obj); }
        PdObj::PdBlock(bb) => {
            bb.run(env);
        }
    }
}

fn initialize(env: &mut Environment) {
    let plus_case: Rc<dyn Case> = Rc::new(BinaryIntCase { func: |_, a, b| vec![Rc::new(PdObj::PdInt(a + b))] });
    let minus_case: Rc<dyn Case> = Rc::new(BinaryIntCase { func: |_, a, b| vec![Rc::new(PdObj::PdInt(a - b))] });

    let mut add_cases = |name: &str, cases: Vec<Rc<dyn Case>>| {
        env.variables.insert(name.to_string(), Rc::new(PdObj::PdBlock(Box::new(CasedBuiltIn {
            name: name.to_string(),
            cases,
        }))));
    };

    add_cases("+", vec![Rc::clone(&plus_case)]);
    add_cases("-", vec![Rc::clone(&minus_case)]);
    add_cases("Test", vec![Rc::clone(&plus_case), Rc::clone(&minus_case)]);

    env.variables.insert("X".to_string(), Rc::new(PdObj::PdInt(3.to_bigint().unwrap())));
}

fn main() {
    let code = "3 4+5+6-";
    let mut env = Environment {
        stack: Vec::new(),
        x_stack: Vec::new(),
        variables: HashMap::new(),
    };

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
                        env.stack.push(Rc::new(PdObj::PdInt(x.clone())));
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
                println!("{:?}", &code[i..j].parse::<BigInt>());
                i = j;
            },
            None => {
                go = false;
            },
        }
    }

    println!("{:?}", env);
}
