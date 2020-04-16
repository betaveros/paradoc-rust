use std::rc::Rc;
use num::bigint::ToBigInt;
extern crate paradoc;

fn int(x: i32) -> Rc<paradoc::PdObj> {
    Rc::new(paradoc::PdObj::PdInt(x.to_bigint().unwrap()))
}

#[test]
fn basic() {
    assert_eq!(paradoc::simple_eval("3 4+"), vec![int(7)]);
    assert_eq!(paradoc::simple_eval("3:"), vec![int(3), int(3)]);
    assert_eq!(paradoc::simple_eval("11 7%"), vec![int(4)]);
}

#[test]
fn list() {
    assert_eq!(paradoc::simple_eval("[3 4]~+"), vec![int(7)]);
}

#[test]
fn map() {
    assert_eq!(paradoc::simple_eval("[3 4])m"), vec![Rc::new(paradoc::PdObj::PdList(vec![int(4), int(5)]))]);
}

#[test]
fn block() {
    assert_eq!(paradoc::simple_eval("3 4{+}~"), vec![int(7)]);
}

#[test]
fn each() {
    assert_eq!(paradoc::simple_eval("[3 4]{:}e"), vec![int(3), int(3), int(4), int(4)]);
    assert_eq!(paradoc::simple_eval("[3 4]:e"), vec![int(3), int(3), int(4), int(4)]);
    assert_eq!(paradoc::simple_eval("[3 4]{2*1+}e"), vec![int(7), int(9)]);
}
