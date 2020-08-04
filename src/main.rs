use std::io;
use std::io::Write;
use paradoc::Block;

fn prompt(input: &mut String) -> bool {
    input.clear();
    print!("prdc-rs> ");
    io::stdout().flush();

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

        block.run(&mut env);

        println!("{}", env.stack_to_repr_string());
    }
}
