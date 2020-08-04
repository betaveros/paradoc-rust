use std::cmp::Ordering;
use std::ops::{Add, Sub, Mul, Div, Rem};
use std::ops::AddAssign;
use std::iter::Sum;
use std::mem;
use num::Integer;
use num::bigint::BigInt;
use num::bigint::ToBigInt;
use num_traits::cast::ToPrimitive;

#[derive(Debug, Clone)]
pub enum PdNum {
    Int(BigInt),
    Float(f64),
    Char(BigInt),
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
            PdNum::Int(n) => self.clone(),
            PdNum::Char(c) => self.clone(),
            PdNum::Float(f) => PdNum::Int(f.ceil().to_bigint().expect("Ceiling of float was not integer")),
        }
    }

    pub fn floor(&self) -> PdNum {
        match self {
            PdNum::Int(n) => self.clone(),
            PdNum::Char(c) => self.clone(),
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
            _ => false,
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
            _ => None,
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

macro_rules! impl_binary_method {
    ($imp:ident, $method:ident, $intmethod:expr, $floatmethod:expr) => {
        impl $imp<&PdNum> for &PdNum {
            type Output = PdNum;

            def_binary_method!($method, $intmethod, $floatmethod);
        }
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

impl Add<PdNum> for PdNum {
    type Output = PdNum;

    fn add(self, other: PdNum) -> PdNum { (&self).add(&other) }
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
