use std::rc::Rc;
// use num::bigint::BigInt;
extern crate paradoc;
use paradoc::PdObj;
use num::complex::Complex64;

fn int(x: i32) -> PdObj {
    PdObj::from(x)
}

fn list(xs: Vec<PdObj>) -> PdObj {
    PdObj::List(Rc::new(xs))
}


macro_rules! intvec {
    ($($case:expr),*) => {
        vec![$( int($case) ),*];
    }
}

macro_rules! lv {
    ($($case:expr),*) => {
        list(vec![$( $case ),*]);
    }
}

macro_rules! liv {
    ($($case:expr),*) => {
        list(intvec![$( $case ),*]);
    }
}

#[test]
fn basic() {
    assert_eq!(paradoc::simple_eval("3 4+"), intvec![7]);
    assert_eq!(paradoc::simple_eval("3:"), intvec![3, 3]);
    assert_eq!(paradoc::simple_eval("11 7%"), intvec![4]);
}

#[test]
fn math_arithmetic() {
    assert_eq!(paradoc::simple_eval("3 4+"), intvec![7]);
    assert_eq!(paradoc::simple_eval("3 4-"), intvec![-1]);
    assert_eq!(paradoc::simple_eval("3 4*"), intvec![12]);
    assert_eq!(paradoc::simple_eval("3 4/"), vec![PdObj::from(0.75f64)]);
}

#[test]
fn math_binary_operations() {
    assert_eq!(paradoc::simple_eval("3 5&"), intvec![1]);
    assert_eq!(paradoc::simple_eval("3 5|"), intvec![7]);
    assert_eq!(paradoc::simple_eval("3 5^"), intvec![6]);
    assert_eq!(paradoc::simple_eval("3 5<s"), intvec![96]);
    assert_eq!(paradoc::simple_eval("253 492&"), intvec![236]);
    assert_eq!(paradoc::simple_eval("253 492|"), intvec![509]);
    assert_eq!(paradoc::simple_eval("253 492^"), intvec![273]);
    assert_eq!(paradoc::simple_eval("253 3<s"), intvec![2024]);
    assert_eq!(paradoc::simple_eval("253 3>s"), intvec![31]);
}

#[test]
fn math_complex() {
    assert_eq!(paradoc::simple_eval("1j 1j*"), vec![PdObj::from(Complex64::from(-1.0))]);
    assert_eq!(paradoc::simple_eval("3 4+j"), vec![PdObj::from(Complex64::new(3.0, 4.0))]);
    assert_eq!(paradoc::simple_eval("[3 4]Rj"), vec![PdObj::from(Complex64::new(3.0, 4.0))]);
    assert_eq!(paradoc::simple_eval("3 4j+Aj"), vec![list(vec![PdObj::from(3.0), PdObj::from(4.0)])]);
}

#[test]
fn readme() {
    assert_eq!(paradoc::simple_eval("¹²m"), vec![liv![0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100]]);
}

#[test]
fn test_list() {
    assert_eq!(paradoc::simple_eval("[3 4]~+"), vec![int(7)]);
}

#[test]
fn parity() {
    assert_eq!(paradoc::simple_eval("3 Od 4 Od 5 Ô 6 Ô"), intvec![1, 0, 1, 0]);
    assert_eq!(paradoc::simple_eval("3 Ev 4 Ev 5 Ê 6 Ê"), intvec![0, 1, 0, 1]);
}

#[test]
fn bool_not() {
    assert_eq!(paradoc::simple_eval("0! 1!"), intvec![1, 0]);
}

#[test]
fn map() {
    assert_eq!(paradoc::simple_eval("[3 4])m"), vec![liv![4, 5]]);
    assert_eq!(paradoc::simple_eval("[3 4]{Y+}%"), vec![liv![3, 5]]);
}

#[test]
fn block() {
    assert_eq!(paradoc::simple_eval("3 4{+}~"), vec![int(7)]);
}

#[test]
fn composition() {
    assert_eq!(paradoc::simple_eval("253 492 {^} {²} + ~"), intvec![74529]);
}

#[test]
fn short_block() {
    assert_eq!(paradoc::simple_eval("[3 4]βm))"), vec![liv![5, 6]]);
    assert_eq!(paradoc::simple_eval("[3 4]γm)))"), vec![liv![6, 7]]);
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
    assert_eq!(paradoc::simple_eval("3 4:o"), intvec![3, 4, 3]);
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
    assert_eq!(paradoc::simple_eval("[3 7 2 5]1<q>"), vec![liv![3], liv![7, 2, 5]]);
}

#[test]
fn looping() {
    assert_eq!(paradoc::simple_eval("[2 5 3])e"), intvec![3, 6, 4]);
    assert_eq!(paradoc::simple_eval("[2 5 3])m"), vec![liv![3, 6, 4]]);
    assert_eq!(paradoc::simple_eval("[2 5 3 6 1]{4<}f"), vec![liv![2, 3, 1]]);
    assert_eq!(paradoc::simple_eval("[2 5 3]3-v"), vec![liv![-1, 2, 0]]);
    assert_eq!(paradoc::simple_eval("9[2 5 3]-y"), vec![liv![7, 4, 6]]);
}

#[test]
fn quantifiers() {
    assert_eq!(paradoc::simple_eval("[2 4 6]Odê"), intvec![0]);
    assert_eq!(paradoc::simple_eval("[2 5 3]Odê"), intvec![1]);
    assert_eq!(paradoc::simple_eval("[3 5 7]Odê"), intvec![1]);
    assert_eq!(paradoc::simple_eval("[2 4 6]Odâ"), intvec![0]);
    assert_eq!(paradoc::simple_eval("[2 5 3]Odâ"), intvec![0]);
    assert_eq!(paradoc::simple_eval("[3 5 7]Odâ"), intvec![1]);
    assert_eq!(paradoc::simple_eval("[2 4 6]Odô"), intvec![1]);
    assert_eq!(paradoc::simple_eval("[2 5 3]Odô"), intvec![0]);
    assert_eq!(paradoc::simple_eval("[3 5 7]Odô"), intvec![0]);
}

#[test]
fn organize() {
    assert_eq!(paradoc::simple_eval("5{2%}ø"), vec![lv![liv![0, 2, 4], liv![1, 3]]]);
}

#[test]
fn sort() {
    assert_eq!(paradoc::simple_eval("[3 1 4 1 5]$"), vec![liv![1, 1, 3, 4, 5]]);
    assert_eq!(paradoc::simple_eval("[3 1 4 1 5]M_$"), vec![liv![5, 4, 3, 1, 1]]);
}

#[test]
fn products() {
    assert_eq!(paradoc::simple_eval("[2 4][6 0 1]*"), vec![
        lv![
            liv![2, 6], liv![2, 0], liv![2, 1],
            liv![4, 6], liv![4, 0], liv![4, 1]
        ]
    ]);
    assert_eq!(paradoc::simple_eval("[2 4][6 0 1]T"), vec![
        lv![
            lv![liv![2, 6], liv![2, 0], liv![2, 1]],
            lv![liv![4, 6], liv![4, 0], liv![4, 1]]
        ]
    ]);
    assert_eq!(paradoc::simple_eval("[2 4][6 0 1]+_B"), vec![
        liv![8, 2, 3, 10, 4, 5]
    ]);
    assert_eq!(paradoc::simple_eval("[2 4][6 0 1]+_T"), vec![
        lv![
            liv![8, 2, 3],
            liv![10, 4, 5]
        ]
    ]);
    assert_eq!(paradoc::simple_eval("[2 3]²"), vec![
        lv![
            lv![liv![2, 2], liv![2, 3]],
            lv![liv![3, 2], liv![3, 3]]
        ]
    ]);
    assert_eq!(paradoc::simple_eval("[2 3]³"), vec![
         lv![
            lv![
                lv![liv![2, 2, 2], liv![2, 2, 3]],
                lv![liv![2, 3, 2], liv![2, 3, 3]]
            ],
            lv![
                lv![liv![3, 2, 2], liv![3, 2, 3]],
                lv![liv![3, 3, 2], liv![3, 3, 3]]
            ]
         ]
    ]);
}

#[test]
fn transpose() {
    assert_eq!(paradoc::simple_eval("[[1][2 3 4][5 6]]™"), vec![lv![liv![1, 2, 5], liv![3, 6], liv![4]]]);
}

#[test]
fn is_sorted_by() {
    assert_eq!(paradoc::simple_eval("[1 3 5]$p"), intvec![1]);
    assert_eq!(paradoc::simple_eval("[2 5 3]$p"), intvec![0]);
    assert_eq!(paradoc::simple_eval("[4 4 8]$p"), intvec![1]);
    assert_eq!(paradoc::simple_eval("[5 3 1]$p"), intvec![0]);
    assert_eq!(paradoc::simple_eval("[1 3 5]<p"), intvec![1]);
    assert_eq!(paradoc::simple_eval("[2 5 3]<p"), intvec![0]);
    assert_eq!(paradoc::simple_eval("[4 4 8]<p"), intvec![0]);
    assert_eq!(paradoc::simple_eval("[5 3 1]<p"), intvec![0]);
    assert_eq!(paradoc::simple_eval("[1 3 5]>p"), intvec![0]);
    assert_eq!(paradoc::simple_eval("[2 5 3]>p"), intvec![0]);
    assert_eq!(paradoc::simple_eval("[4 4 8]>p"), intvec![0]);
    assert_eq!(paradoc::simple_eval("[5 3 1]>p"), intvec![1]);
}

#[test]
fn hoard() {
    assert_eq!(paradoc::simple_eval("1 2 Hu 4 5 Hu 4 H="), intvec![5]);
}

#[test]
fn split() {
    assert_eq!(paradoc::simple_eval("[1 2 3]2/"), vec![lv![liv![1, 2], liv![3]]]);
    assert_eq!(paradoc::simple_eval("[1 2 3 4]2/"), vec![lv![liv![1, 2], liv![3, 4]]]);
    assert_eq!(paradoc::simple_eval("[1 2 3]2÷"), vec![lv![liv![1, 2]]]);
    assert_eq!(paradoc::simple_eval("[1 2 3 4]2÷"), vec![lv![liv![1, 2], liv![3, 4]]]);
    assert_eq!(paradoc::simple_eval("\"12345\"2/"), vec![lv![PdObj::from("12"), PdObj::from("34"), PdObj::from("5")]]);
}

#[test]
fn join() {
    assert_eq!(paradoc::simple_eval(r#""abc""."R"#), vec![PdObj::from("a.b.c")]);
    assert_eq!(paradoc::simple_eval(r#"["ab" "cd" "ef"]","R"#), vec![PdObj::from("ab,cd,ef")]);
    assert_eq!(paradoc::simple_eval(r#"[2 5 3]0R"#), vec![liv![2, 0, 5, 0, 3]]);
    assert_eq!(paradoc::simple_eval(r#"[24 60 1]","R"#), vec![PdObj::from("24,60,1")]);
    assert_eq!(paradoc::simple_eval(r#"[24 60 1]',R"#), vec![PdObj::from("24,60,1")]);
}

#[test]
fn reduce() {
    assert_eq!(paradoc::simple_eval(r"10,{+}R"), vec![int(45)]);
    assert_eq!(paradoc::simple_eval(r"[2 5 3]{+}R"), vec![int(10)]);
    assert_eq!(paradoc::simple_eval(r"10,+r"), vec![int(45)]);
    assert_eq!(paradoc::simple_eval(r"[5 17 20]|r"), vec![int(21)]);
}

#[test]
fn base_conversion() {
    assert_eq!(paradoc::simple_eval("[2 0 1]3B"), intvec![19]);
    assert_eq!(paradoc::simple_eval("[2m 0 1m]3B"), intvec![-19]);
    assert_eq!(paradoc::simple_eval("19 3B"), vec![liv![2, 0, 1]]);
    assert_eq!(paradoc::simple_eval("19m 3B"), vec![liv![-2, 0, -1]]);
}

#[test]
fn type_predicates() {
    assert_eq!(paradoc::simple_eval("[2 0 1]3B"), intvec![19]);
    assert_eq!(
        paradoc::simple_eval("[1 1.0 '1 \"1\" [1] {1} H][:i_:f_:c_:n_:s_:a_:b_:h_],y"),
        vec![lv![
            liv![0],
            liv![1],
            liv![2],
            liv![0,1,2],
            liv![3],
            liv![4],
            liv![5],
            liv![6]
        ]]
    )
}
