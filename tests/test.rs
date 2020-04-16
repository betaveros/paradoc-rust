use std::rc::Rc;
use num::bigint::ToBigInt;
extern crate paradoc;

fn int(x: i32) -> Rc<paradoc::PdObj> {
    Rc::new(paradoc::PdObj::PdInt(x.to_bigint().unwrap()))
}

#[test]
fn basic() {
    assert_eq!(paradoc::simple_eval("3 4+"), vec![int(7)]);
    assert_eq!(paradoc::simple_eval("11 7%"), vec![int(4)]);
}

#[test]
fn list() {
    assert_eq!(paradoc::simple_eval("[3 4]~+"), vec![int(7)]);
}

#[test]
fn block() {
    assert_eq!(paradoc::simple_eval("3 4{+}~"), vec![int(7)]);
}
