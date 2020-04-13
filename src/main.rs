fn main() {
    // let code = "3 4+5+6-X+";
    let code = "3 4+:*";
    println!("{:?}", paradoc::simple_eval(code));
}
