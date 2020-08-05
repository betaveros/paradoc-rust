use std::io;
use std::io::Write;
use paradoc::Block;

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

fn main() {
    let mut env = paradoc::Environment::new();
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
