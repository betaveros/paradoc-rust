use std::rc::Rc;
// use num::bigint::BigInt;
extern crate paradoc;

fn int(x: i32) -> paradoc::PdObj {
    paradoc::PdObj::from(x)
}

fn list(xs: Vec<paradoc::PdObj>) -> paradoc::PdObj {
    paradoc::PdObj::List(Rc::new(xs))
}


macro_rules! intvec {
    ($($case:expr),*) => {
        vec![$( int($case), )*];
    }
}

#[test]
fn basic() {
    assert_eq!(paradoc::simple_eval("3 4+"), intvec![7]);
    assert_eq!(paradoc::simple_eval("3:"), intvec![3, 3]);
    assert_eq!(paradoc::simple_eval("11 7%"), intvec![4]);
}

#[test]
fn readme() {
    assert_eq!(paradoc::simple_eval("¹²m"), vec![list(intvec![0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100])]);
}

#[test]
fn test_list() {
    assert_eq!(paradoc::simple_eval("[3 4]~+"), vec![int(7)]);
}

#[test]
fn map() {
    assert_eq!(paradoc::simple_eval("[3 4])m"), vec![list(intvec![4, 5])]);
    assert_eq!(paradoc::simple_eval("[3 4]{Y+}%"), vec![list(intvec![3, 5])]);
}

#[test]
fn block() {
    assert_eq!(paradoc::simple_eval("3 4{+}~"), vec![int(7)]);
}

#[test]
fn each() {
    assert_eq!(paradoc::simple_eval("[3 4]{:}e"), intvec![3, 3, 4, 4]);
    assert_eq!(paradoc::simple_eval("[3 4]:e"), intvec![3, 3, 4, 4]);
    assert_eq!(paradoc::simple_eval("[3 4]{2*1+}e"), intvec![7, 9]);
}

#[test]
fn stack_manip() {
    assert_eq!(paradoc::simple_eval("3 4:" ), intvec![3, 4, 4]);
    assert_eq!(paradoc::simple_eval("3 4:p"), intvec![3, 4, 3, 4]);
    assert_eq!(paradoc::simple_eval("3 4:a"), intvec![3, 4, 3]);
    assert_eq!(paradoc::simple_eval("3 4\\"), intvec![4, 3]);

    assert_eq!(paradoc::simple_eval("3 4 5\\o"), intvec![4, 5, 3]);
    assert_eq!(paradoc::simple_eval("3 4 5\\i"), intvec![5, 3, 4]);
}

#[test]
fn stack_manip_trailer() {
    assert_eq!(paradoc::simple_eval("3 4 5+u" ), intvec![7, 5]);
    assert_eq!(paradoc::simple_eval("3 4 5+k" ), intvec![3, 4, 5, 9]);
    assert_eq!(paradoc::simple_eval("3 4 5+q" ), intvec![3, 9, 4, 5]);
}

#[test]
fn comparison() {
    assert_eq!(paradoc::simple_eval("3 3= 3 4= 4 3=" ), intvec![1, 0, 0]);
    assert_eq!(paradoc::simple_eval("3 3< 3 4< 4 3<" ), intvec![0, 1, 0]);
    assert_eq!(paradoc::simple_eval("3 3> 3 4> 4 3>" ), intvec![0, 0, 1]);
}

#[test]
fn indexing() {
    assert_eq!(paradoc::simple_eval("[3 1 4 1 5 9] 0=q;2=q;5=q;1m=q;6m="), intvec![3, 4, 9, 9, 3]);
}
#[test]
fn slicing() {
    assert_eq!(paradoc::simple_eval("[3 7 2 5]1<q>"), vec![list(intvec![3]), list(intvec![7, 2, 5])]);
}

#[test]
fn looping() {
    assert_eq!(paradoc::simple_eval("[2 5 3])e"), intvec![3, 6, 4]);
    assert_eq!(paradoc::simple_eval("[2 5 3])m"), vec![list(intvec![3, 6, 4])]);
    assert_eq!(paradoc::simple_eval("[2 5 3 6 1]{4<}f"), vec![list(intvec![2, 3, 1])]);
    assert_eq!(paradoc::simple_eval("[2 5 3]3-v"), vec![list(intvec![-1, 2, 0])]);
    assert_eq!(paradoc::simple_eval("9[2 5 3]-y"), vec![list(intvec![7, 4, 6])]);
}

#[test]
fn organize() {
    assert_eq!(paradoc::simple_eval("5{2%}ø"), vec![list(vec![list(intvec![0, 2, 4]), list(intvec![1, 3])])]);
}
