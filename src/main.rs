use std::io;
use std::fs::File;
use std::io::{Read, Write};
use paradoc::Block;
use paradoc::pderror::PdError;

fn prompt(input: &mut String) -> bool {
    input.clear();
    print!("prdc-rs> ");
    if io::stdout().flush().is_err() { return false }

    match io::stdin().read_line(input) {
        Ok(0) => false,
        Ok(_) => true,
        Err(_) => false,
    }
}

fn repl() {
    let mut env = paradoc::Environment::new_with_stdio();
    paradoc::initialize(&mut env);

    let mut input = String::new();
    while prompt(&mut input) {
        let block = paradoc::CodeBlock::parse(&input);

        match block.run(&mut env) {
            Ok(()) => {},
            Err(e) => { println!("ERROR: {:?}", e); }
        }

        println!("{}", env.stack_to_repr_string());
    }
}

fn main() {
    match std::env::args().collect::<Vec<String>>().as_slice() {
        [] | [_] => { repl(); }
        [_, s] => {
            match File::open(s) {
                Ok(mut file) => {
                    let mut code = String::new();
                    file.read_to_string(&mut code).expect("reading code file failed");

                    let block = paradoc::CodeBlock::parse(&code);

                    // println!("{:?}", block);

                    let mut env = paradoc::Environment::new_with_stdio();
                    paradoc::initialize(&mut env);

                    match block.run(&mut env) {
                        Ok(()) => { println!("{}", env.stack_to_string()) }
                        Err(PdError::Exit) => {}
                        Err(e) => { println!("ERROR: {:?}", e); }
                    }
                }
                Err(_) => {
                    panic!("opening code file failed");
                }
            }
        }
        _ => { panic!("too many arguments"); }
    }
}
