use std::cmp::Ordering;
use std::ops::{Add, Sub, Mul, Div, Rem, Neg};
use std::ops::AddAssign;
use std::iter::{Sum, Product};
use std::mem;
use num::Integer;
use num::bigint::BigInt;
use num::bigint::ToBigInt;
use num_traits::pow::Pow;
use num_traits::cast::ToPrimitive;

#[derive(Debug, Clone)]
pub enum PdNum {
    Int(BigInt),
    Float(f64),
    Char(BigInt),
}

impl From<BigInt> for PdNum {
    fn from(x: BigInt) -> Self { PdNum::Int(x) }
}
impl From<char> for PdNum {
    fn from(c: char) -> Self { PdNum::Char(BigInt::from(c as u32)) }
}
impl From<i32> for PdNum {
    fn from(x: i32) -> Self { PdNum::Int(BigInt::from(x)) }
}
impl From<f64> for PdNum {
    fn from(x: f64) -> Self { PdNum::Float(x) }
}
impl From<usize> for PdNum {
    fn from(x: usize) -> Self { PdNum::Int(BigInt::from(x)) }
}

impl PdNum {
    pub fn to_string(&self) -> String {
        match self {
            PdNum::Int(n) => n.to_string(),
            PdNum::Float(f) => f.to_string(),
            // TODO: handle gracefully
            // "and_then" is >>=
            PdNum::Char(c) => c.to_u32().and_then(std::char::from_u32).map_or(c.to_string(), |x| x.to_string()),
        }
    }

    pub fn ceil(&self) -> PdNum {
        match self {
            PdNum::Int(_) => self.clone(),
            PdNum::Char(_) => self.clone(),
            PdNum::Float(f) => PdNum::Int(f.ceil().to_bigint().expect("Ceiling of float was not integer")),
        }
    }

    pub fn floor(&self) -> PdNum {
        match self {
            PdNum::Int(_) => self.clone(),
            PdNum::Char(_) => self.clone(),
            PdNum::Float(f) => PdNum::Int(f.floor().to_bigint().expect("Floor of float was not integer")),
        }
    }

    pub fn add_const(&self, k: i32) -> PdNum {
        match self {
            PdNum::Int(n) => PdNum::Int(n + k),
            PdNum::Char(c) => PdNum::Char(c + k),
            PdNum::Float(f) => PdNum::Float(f + (k as f64)),
        }
    }

    pub fn is_nonzero(&self) -> bool {
        match self {
            PdNum::Int(i) => *i != BigInt::from(0),
            PdNum::Float(f) => *f != 0.0,
            PdNum::Char(c) => *c != BigInt::from(0),
        }
    }

    pub fn to_f64(&self) -> Option<f64> {
        match self {
            PdNum::Int(i) => i.to_f64(),
            PdNum::Float(f) => Some(*f),
            PdNum::Char(c) => c.to_f64(),
        }
    }

    pub fn sqrt(&self) -> Option<PdNum> {
        self.to_f64().map(|x| PdNum::Float(x.sqrt()))
    }

    pub fn pow(&self, e: u32) -> PdNum {
        match self {
            PdNum::Int(i) => PdNum::Int(i.pow(e)),
            PdNum::Float(f) => PdNum::Float(f.powi(e as i32)),
            PdNum::Char(c) => PdNum::Char(c.pow(e)),
        }
    }

    pub fn is_nan(&self) -> bool {
        match self {
            PdNum::Int(_) => false,
            PdNum::Float(f) => f.is_nan(),
            PdNum::Char(_) => false,
        }
    }
}

// this seems... nontrivial??
fn cmp_bigint_f64(a: &BigInt, b: &f64) -> Option<Ordering> {
    if let Some(bi) = b.to_bigint() {
        Some(a.cmp(&bi))
    } else {
        b.floor().to_bigint().map(|bi| {
            match a.cmp(&bi) {
                Ordering::Less    => Ordering::Less,
                Ordering::Equal   => Ordering::Less,
                Ordering::Greater => Ordering::Greater,
            }
        })
    }
}

impl PartialEq for PdNum {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PdNum::Int   (a), PdNum::Int   (b)) => a == b,
            (PdNum::Int   (a), PdNum::Float (b)) => b.to_bigint().map_or(false, |x| &x == a),
            (PdNum::Int   (a), PdNum::Char  (b)) => a == b,
            (PdNum::Float (a), PdNum::Int   (b)) => a.to_bigint().map_or(false, |x| &x == b),
            (PdNum::Float (a), PdNum::Float (b)) => a == b,
            (PdNum::Float (a), PdNum::Char  (b)) => a.to_bigint().map_or(false, |x| &x == b),
            (PdNum::Char  (a), PdNum::Int   (b)) => a == b,
            (PdNum::Char  (a), PdNum::Float (b)) => b.to_bigint().map_or(false, |x| &x == a),
            (PdNum::Char  (a), PdNum::Char  (b)) => a == b,
        }
    }
}

// TODO: Watch https://github.com/rust-lang/rust/issues/72599, we will probably want total
// orderings in some cases.

impl PartialOrd for PdNum {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (PdNum::Int   (a), PdNum::Int   (b)) => Some(a.cmp(b)),
            (PdNum::Int   (a), PdNum::Float (b)) => cmp_bigint_f64(a, b),
            (PdNum::Int   (a), PdNum::Char  (b)) => Some(a.cmp(b)),
            (PdNum::Float (a), PdNum::Int   (b)) => cmp_bigint_f64(b, a).map(|ord| ord.reverse()),
            (PdNum::Float (a), PdNum::Float (b)) => a.partial_cmp(b),
            (PdNum::Float (a), PdNum::Char  (b)) => cmp_bigint_f64(b, a).map(|ord| ord.reverse()),
            (PdNum::Char  (a), PdNum::Int   (b)) => Some(a.cmp(b)),
            (PdNum::Char  (a), PdNum::Float (b)) => cmp_bigint_f64(a, b),
            (PdNum::Char  (a), PdNum::Char  (b)) => Some(a.cmp(b)),
        }
    }
}

impl PdNum {
    // (considers NaNs equal)
    fn total_cmp_small_nan(&self, other: &Self) -> Ordering {
        match (self, other) {
            (PdNum::Int   (a), PdNum::Int   (b)) => a.cmp(b),
            (PdNum::Int   (a), PdNum::Float (b)) => cmp_bigint_f64(a, b).unwrap_or(Ordering::Greater),
            (PdNum::Int   (a), PdNum::Char  (b)) => a.cmp(b),
            (PdNum::Float (a), PdNum::Int   (b)) => cmp_bigint_f64(b, a).map_or(Ordering::Less, |ord| ord.reverse()),
            (PdNum::Float (a), PdNum::Float (b)) => a.partial_cmp(b).unwrap_or(b.is_nan().cmp(&a.is_nan())), // note swap
            (PdNum::Float (a), PdNum::Char  (b)) => cmp_bigint_f64(b, a).map_or(Ordering::Less, |ord| ord.reverse()),
            (PdNum::Char  (a), PdNum::Int   (b)) => a.cmp(b),
            (PdNum::Char  (a), PdNum::Float (b)) => cmp_bigint_f64(a, b).unwrap_or(Ordering::Greater),
            (PdNum::Char  (a), PdNum::Char  (b)) => a.cmp(b),
        }
    }

    fn total_cmp_big_nan(&self, other: &Self) -> Ordering {
        match (self, other) {
            (PdNum::Int   (a), PdNum::Int   (b)) => a.cmp(b),
            (PdNum::Int   (a), PdNum::Float (b)) => cmp_bigint_f64(a, b).unwrap_or(Ordering::Less),
            (PdNum::Int   (a), PdNum::Char  (b)) => a.cmp(b),
            (PdNum::Float (a), PdNum::Int   (b)) => cmp_bigint_f64(b, a).map_or(Ordering::Greater, |ord| ord.reverse()),
            (PdNum::Float (a), PdNum::Float (b)) => a.partial_cmp(b).unwrap_or(a.is_nan().cmp(&b.is_nan())),
            (PdNum::Float (a), PdNum::Char  (b)) => cmp_bigint_f64(b, a).map_or(Ordering::Greater, |ord| ord.reverse()),
            (PdNum::Char  (a), PdNum::Int   (b)) => a.cmp(b),
            (PdNum::Char  (a), PdNum::Float (b)) => cmp_bigint_f64(a, b).unwrap_or(Ordering::Less),
            (PdNum::Char  (a), PdNum::Char  (b)) => a.cmp(b),
        }
    }
}

// https://github.com/rust-lang/rust/pull/64047 will give us these for free
// note that we follow the Rust implementations and in particular the f64 implementations of min
// and max: when equal, pretend the left is smaller than the right; if one of its inputs is NaN,
// return the other

impl PdNum {
    pub fn min<'a>(&'a self, other: &'a Self) -> &'a PdNum {
        match self.total_cmp_big_nan(other) {
            Ordering::Greater => other,
            _ => self,
        }
    }

    pub fn max<'a>(&'a self, other: &'a Self) -> &'a PdNum {
        match self.total_cmp_small_nan(other) {
            Ordering::Less => self,
            _ => other,
        }
    }
}

//// ????????
///
macro_rules! binary_match {
    ($a:expr, $b:expr, $method:ident, $intmethod:expr, $floatmethod:expr) => {
        match ($a, $b) {
            (PdNum::Int  (a), PdNum::Int  (b)) => PdNum::Int($intmethod(a, b)),
            (PdNum::Int  (a), PdNum::Float(b)) => PdNum::Float($floatmethod(a.to_f64().expect(concat!("num ", stringify!(method), " float halp")), *b)),
            (PdNum::Int  (a), PdNum::Char (b)) => PdNum::Int($intmethod(a, b)),
            (PdNum::Float(a), PdNum::Int  (b)) => PdNum::Float($floatmethod(*a, b.to_f64().expect(concat!("num ", stringify!(method), " float halp")))),
            (PdNum::Float(a), PdNum::Float(b)) => PdNum::Float($floatmethod(*a, *b)),
            (PdNum::Float(a), PdNum::Char (b)) => PdNum::Float($floatmethod(*a, b.to_f64().expect(concat!("num ", stringify!(method), " float halp")))),
            (PdNum::Char (a), PdNum::Int  (b)) => PdNum::Int($intmethod(a, b)),
            (PdNum::Char (a), PdNum::Float(b)) => PdNum::Float($floatmethod(a.to_f64().expect(concat!("num ", stringify!(method), " float halp")), *b)),
            (PdNum::Char (a), PdNum::Char (b)) => PdNum::Char($intmethod(a, b)),
        }
    };
}

macro_rules! def_binary_method {
    ($method:ident, $intmethod:expr, $floatmethod:expr) => {
        fn $method(self, other: &PdNum) -> PdNum {
            binary_match!(self, other, $method, $intmethod, $floatmethod)
        }
    };
}

macro_rules! forward_impl_binary_method {
    ($imp:ident, $method:ident) => {
        impl $imp<PdNum> for PdNum {
            type Output = PdNum;

            fn $method(self, other: PdNum) -> PdNum { (&self).$method(&other) }
        }
    };
}

macro_rules! impl_binary_method {
    ($imp:ident, $method:ident, $intmethod:expr, $floatmethod:expr) => {
        impl $imp<&PdNum> for &PdNum {
            type Output = PdNum;

            def_binary_method!($method, $intmethod, $floatmethod);
        }

        forward_impl_binary_method!($imp, $method);
    };
}

impl PdNum {
    pub fn div_floor(&self, other: &PdNum) -> PdNum {
        binary_match!(self, other, div_floor, Integer::div_floor, f64::div_euclid)
    }
}

impl_binary_method!(Add, add, Add::add, Add::add);
impl_binary_method!(Sub, sub, Sub::sub, Sub::sub);
impl_binary_method!(Mul, mul, Mul::mul, Mul::mul);
impl_binary_method!(Rem, rem, Integer::mod_floor, f64::rem_euclid);

impl Div<&PdNum> for &PdNum {
    type Output = PdNum;
    fn div(self, other: &PdNum) -> PdNum {
        PdNum::Float(self.to_f64().expect("division float fail") / other.to_f64().expect("division float fail"))
    }
}

forward_impl_binary_method!(Div, div);

impl Neg for PdNum {
    type Output = PdNum;

    fn neg(self) -> PdNum {
        match self {
            PdNum::Int(n) => PdNum::Int(-n),
            PdNum::Float(f) => PdNum::Float(-f),
            PdNum::Char(c) => PdNum::Char(-c),
        }
    }
}

impl AddAssign<&PdNum> for PdNum {
    fn add_assign(&mut self, other: &PdNum) {
        let n = mem::replace(self, PdNum::Int(BigInt::from(0)));
        *self = &n + other;
    }
}

impl Sum for PdNum {
    fn sum<I: Iterator<Item=Self>>(iter: I) -> Self {
        iter.fold(PdNum::Int(BigInt::from(0)), Add::add)
    }
}

impl Product for PdNum {
    fn product<I: Iterator<Item=Self>>(iter: I) -> Self {
        iter.fold(PdNum::Int(BigInt::from(1)), Mul::mul)
    }
}
