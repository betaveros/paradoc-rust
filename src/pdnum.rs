use std::rc::Rc;
use std::cmp::Ordering;
use std::ops::{Add, Sub, Mul, Div, Rem, BitAnd, BitOr, BitXor, Neg, Deref};
use std::ops::AddAssign;
use std::iter::{Sum, Product};
use std::hash::{Hash, Hasher};
use std::mem;
use num::Integer;
use num::bigint::BigInt;
use num::bigint::ToBigInt;
use num_traits::pow::Pow;
use num_traits::sign::Signed;
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

fn pow_big_ints(a: &BigInt, b: &BigInt) -> PdNum {
    match b.sign() {
        num::bigint::Sign::NoSign => PdNum::from(0),
        num::bigint::Sign::Plus => PdNum::from(Pow::pow(a, b.magnitude())),
        num::bigint::Sign::Minus => PdNum::from(a.to_f64().expect("exponent c'mon").pow(b.to_f64().expect("exponent c'mon"))),
    }
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

    pub fn repr(&self) -> String {
        match self {
            PdNum::Int(n)   => n.to_string(),
            PdNum::Float(f) => f.to_string(),
            // TODO as above
            PdNum::Char(c)  => c.to_u32().and_then(std::char::from_u32).map_or_else(
                || ".'".to_string() + &c.to_string(),
                |ch| ['\'', ch].iter().collect::<String>())
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

    pub fn trunc(&self) -> PdNum {
        match self {
            PdNum::Int(_) => self.clone(),
            PdNum::Char(_) => self.clone(),
            PdNum::Float(f) => PdNum::Int(f.trunc().to_bigint().expect("Truncation of float was not integer")),
        }
    }

    pub fn abs(&self) -> PdNum {
        match self {
            PdNum::Int(k) => PdNum::Int(k.abs()),
            PdNum::Char(k) => PdNum::Char(k.abs()),
            PdNum::Float(f) => PdNum::Float(f.abs()),
        }
    }

    pub fn signum(&self) -> PdNum {
        match self {
            PdNum::Int(k) => PdNum::Int(k.signum()),
            PdNum::Char(k) => PdNum::Char(k.signum()),
            PdNum::Float(f) => {
                // This is NOT Rust's f64's signum. We want +/-0 to give 0 (for consistency with
                // integers)
                if f.is_nan() {
                    PdNum::Float(*f)
                } else if *f == 0.0 {
                    PdNum::from(0)
                } else if *f > 0.0 {
                    PdNum::from(1)
                } else {
                    PdNum::from(-1)
                }
            }
        }
    }

    pub fn add_const(&self, k: i32) -> PdNum {
        match self {
            PdNum::Int(n) => PdNum::Int(n + k),
            PdNum::Char(c) => PdNum::Char(c + k),
            PdNum::Float(f) => PdNum::Float(f + (k as f64)),
        }
    }

    pub fn mul_const(&self, k: i32) -> PdNum {
        match self {
            PdNum::Int(n) => PdNum::Int(n * k),
            PdNum::Char(c) => PdNum::Char(c * k),
            PdNum::Float(f) => PdNum::Float(f * (k as f64)),
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

    pub fn pow_num(&self, other: &PdNum) -> PdNum {
        match (self, other) {
            (PdNum::Int   (a), PdNum::Int   (b)) => pow_big_ints(a, b),
            (PdNum::Int   (a), PdNum::Float (b)) => PdNum::from(a.to_f64().expect("pow pls").pow(b)),
            (PdNum::Int   (a), PdNum::Char  (b)) => pow_big_ints(a, b),
            (PdNum::Float (a), PdNum::Int   (b)) => PdNum::from(a.pow(b.to_f64().expect("pow pls"))),
            (PdNum::Float (a), PdNum::Float (b)) => PdNum::from(a.pow(b)),
            (PdNum::Float (a), PdNum::Char  (b)) => PdNum::from(a.pow(b.to_f64().expect("pow pls"))),
            (PdNum::Char  (a), PdNum::Int   (b)) => pow_big_ints(a, b),
            (PdNum::Char  (a), PdNum::Float (b)) => PdNum::from(a.to_f64().expect("pow pls").pow(b)),
            (PdNum::Char  (a), PdNum::Char  (b)) => pow_big_ints(a, b),
        }
    }

    pub fn is_nan(&self) -> bool {
        match self {
            PdNum::Int(_) => false,
            PdNum::Float(f) => f.is_nan(),
            PdNum::Char(_) => false,
        }
    }

    pub fn to_isize(&self) -> Option<isize> {
        match self {
            PdNum::Int(n) => n.to_isize(),
            PdNum::Float(f) => f.trunc().to_isize(),
            PdNum::Char(c) => c.to_isize(),
        }
    }

    pub fn to_usize(&self) -> Option<usize> {
        match self {
            PdNum::Int(n) => n.to_usize(),
            PdNum::Float(f) => f.trunc().to_usize(),
            PdNum::Char(c) => c.to_usize(),
        }
    }

    pub fn to_char(&self) -> Option<char> {
        std::char::from_u32(match self {
            PdNum::Int(n) => n.to_u32()?,
            PdNum::Float(f) => f.trunc().to_u32()?,
            PdNum::Char(c) => c.to_u32()?,
        })
    }

    pub fn to_clamped_usize(&self) -> usize {
        match self {
            PdNum::Int(n) => {
                if n <= &BigInt::from(0) { 0usize } else { n.to_usize().unwrap_or(usize::MAX) }
            }
            PdNum::Float(f) => {
                if *f <= 0.0 || f.is_nan() { 0usize } else { f.trunc().to_usize().unwrap_or(usize::MAX) }
            }
            PdNum::Char(c) => {
                if c <= &BigInt::from(0) { 0usize } else { c.to_usize().unwrap_or(usize::MAX) }
            }
        }
    }

    pub fn to_nn_usize(&self) -> Option<usize> {
        let s = self.to_usize()?;
        if s == 0 { None } else { Some(s) }
    }

    pub fn to_bigint(&self) -> Option<BigInt> {
        match self {
            PdNum::Int(n) => Some(BigInt::clone(n)),
            PdNum::Float(f) => f.trunc().to_bigint(),
            PdNum::Char(c) => Some(BigInt::clone(c)),
        }
    }

    pub fn construct_like_self(&self, n: BigInt) -> PdNum {
        match self {
            PdNum::Int(_)   => PdNum::Int(n),
            PdNum::Float(_) => PdNum::Int(n),
            PdNum::Char(_)  => PdNum::Char(n),
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


// Tries to follow the laws
#[derive(Debug, Clone)]
pub struct PdTotalNum(pub Rc<PdNum>);

impl Deref for PdTotalNum {
    type Target = Rc<PdNum>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Considers NaNs equal
impl PartialEq for PdTotalNum {
    fn eq(&self, other: &Self) -> bool {
        PdNum::eq(&**self, &**other) || self.is_nan() && other.is_nan()
    }
}

impl Eq for PdTotalNum {}

impl Ord for PdTotalNum {
    fn cmp(&self, other: &Self) -> Ordering {
        self.total_cmp_small_nan(&**other)
    }
}
impl PartialOrd for PdTotalNum {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl Hash for PdTotalNum {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match &***self {
            PdNum::Int(a) => BigInt::hash(a, state),
            PdNum::Char(c) => BigInt::hash(c, state),
            PdNum::Float(f) => match f.to_bigint() {
                Some(s) => BigInt::hash(&s, state),
                None => if f.is_nan() {
                    // some nan from wikipedia (not that this matters)
                    state.write_u64(0x7FF0000000000001u64)
                } else {
                    // I *think* this actually obeys the laws...?
                    // (+/- 0 are handled by the bigint branch)
                    f.to_bits().hash(state)
                }
            }
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
            Ordering::Greater => self,
            _ => other,
        }
    }
}

// ????????
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

    fn neg(self) -> PdNum { -&self }
}
impl Neg for &PdNum {
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

// things that have to be BigInts and we just force it
fn force_bi(f: f64, err: &str) -> BigInt {
    f.trunc().to_bigint().expect(format!("{}: float didn't trunc to integer", err).as_str())
}

macro_rules! force_bi_binary_match {
    ($a:expr, $b:expr, $method:ident, $intmethod:expr) => {
        match ($a, $b) {
            (PdNum::Int  (a), PdNum::Int  (b)) => PdNum::Int($intmethod(a, b)),
            (PdNum::Int  (a), PdNum::Float(b)) => PdNum::Int($intmethod(a, force_bi(*b, stringify!($method)))),
            (PdNum::Int  (a), PdNum::Char (b)) => PdNum::Int($intmethod(a, b)),
            (PdNum::Float(a), PdNum::Int  (b)) => PdNum::Int($intmethod(force_bi(*a, stringify!($method)), b)),
            (PdNum::Float(a), PdNum::Float(b)) => PdNum::Int($intmethod(force_bi(*a, stringify!($method)), force_bi(*b, stringify!($method)))),
            (PdNum::Float(a), PdNum::Char (b)) => PdNum::Int($intmethod(force_bi(*a, stringify!($method)), b)),
            (PdNum::Char (a), PdNum::Int  (b)) => PdNum::Int($intmethod(a, b)),
            (PdNum::Char (a), PdNum::Float(b)) => PdNum::Int($intmethod(a, force_bi(*b, stringify!($method)))),
            (PdNum::Char (a), PdNum::Char (b)) => PdNum::Char($intmethod(a, b)),
        }
    };
}

macro_rules! def_force_bi_binary_method {
    ($method:ident, $intmethod:expr) => {
        fn $method(self, other: &PdNum) -> PdNum {
            force_bi_binary_match!(self, other, $method, $intmethod)
        }
    };
}

macro_rules! impl_force_bi_binary_method {
    ($imp:ident, $method:ident, $intmethod:expr) => {
        impl $imp<&PdNum> for &PdNum {
            type Output = PdNum;

            def_force_bi_binary_method!($method, $intmethod);
        }

        forward_impl_binary_method!($imp, $method);
    };
}

impl_force_bi_binary_method!(BitAnd, bitand, BitAnd::bitand);
impl_force_bi_binary_method!(BitOr, bitor, BitOr::bitor);
impl_force_bi_binary_method!(BitXor, bitxor, BitXor::bitxor);
