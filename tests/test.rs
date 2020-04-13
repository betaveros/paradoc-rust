use std::rc::Rc;
use num::bigint::ToBigInt;
extern crate paradoc;

#[test]
fn basic() {
    assert_eq!(paradoc::simple_eval("3 4+"), vec![Rc::new(paradoc::PdObj::PdInt(7_i32.to_bigint().unwrap()))]);
}

#[test]
fn list() {
    assert_eq!(paradoc::simple_eval("[3 4]~+"), vec![Rc::new(paradoc::PdObj::PdInt(7_i32.to_bigint().unwrap()))]);
}

#[test]
fn block() {
    assert_eq!(paradoc::simple_eval("3 4{+}~"), vec![Rc::new(paradoc::PdObj::PdInt(7_i32.to_bigint().unwrap()))]);
}
