use std::rc::Rc;
use std::cmp::Ordering;
use std::ops::{Add, Sub, Mul, Div, Rem, BitAnd, BitOr, BitXor, Neg, Deref};
use std::ops::AddAssign;
use std::iter::{Sum, Product};
use std::hash::{Hash, Hasher};
use std::mem;
use std::fmt;
use num::Integer;
use num::bigint::BigInt;
use num::bigint::ToBigInt;
use num::complex::Complex64;
use num_traits::pow::Pow;
use num_traits::sign::Signed;
use num_traits::cast::ToPrimitive;
use num_traits::identities::{Zero, One};

use crate::gamma;

#[derive(Debug, Clone)]
pub enum PdNum {
    Int(BigInt),
    Float(f64),
    Char(BigInt),
    Complex(Complex64),
}

impl fmt::Display for PdNum {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PdNum::Int(n) => write!(formatter, "{}", n),
            PdNum::Float(f) => write!(formatter, "{}", f),
            // TODO as above
            PdNum::Char(c)  => match c.to_u32().and_then(std::char::from_u32) {
                Some(ch) => write!(formatter, "{}", ch),
                None => write!(formatter, "\\{}", c),
            }
            PdNum::Complex(z) => write!(formatter, "{}", z),
        }
    }
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
impl From<Complex64> for PdNum {
    fn from(z: Complex64) -> Self { PdNum::Complex(z) }
}

fn pow_big_ints(a: &BigInt, b: &BigInt) -> PdNum {
    match b.sign() {
        num::bigint::Sign::NoSign => PdNum::from(0),
        num::bigint::Sign::Plus => PdNum::from(Pow::pow(a, b.magnitude())),
        num::bigint::Sign::Minus => PdNum::from(a.to_f64().expect("exponent c'mon").pow(b.to_f64().expect("exponent c'mon"))),
    }
}

fn factorial_big_int(a: &BigInt) -> BigInt {
    let mut ret = BigInt::one();
    for i in num_iter::range_inclusive(BigInt::one(), BigInt::clone(a)) {
        ret *= i;
    }
    ret
}

fn bigint_to_f64_or_inf(i: &BigInt) -> f64 {
    i.to_f64().unwrap_or_else(|| {
        if i.is_positive() { f64::INFINITY } else { f64::NEG_INFINITY }
    })
}

macro_rules! forward_int_coercion {
    ($method:ident) => {
        pub fn $method(&self) -> PdNum {
            match self {
                PdNum::Int(_) => self.clone(),
                PdNum::Char(_) => self.clone(),
                PdNum::Float(f) => f.$method().to_bigint().map_or(self.clone(), PdNum::Int),
                PdNum::Complex(z) => z.re.$method().to_bigint().map_or(self.clone(), PdNum::Int),
            }
        }
    };
}

impl PdNum {
    pub fn one_j() -> Self {
        Self::from(Complex64::new(0.0, 1.0))
    }

    pub fn to_string(&self) -> String {
        match self {
            PdNum::Int(n) => n.to_string(),
            PdNum::Float(f) => f.to_string(),
            PdNum::Complex(z) => z.to_string(),
            // TODO: handle gracefully
            // "and_then" is >>=
            PdNum::Char(c) => c.to_u32().and_then(std::char::from_u32).map_or(c.to_string(), |x| x.to_string()),
        }
    }

    pub fn repr(&self) -> String {
        match self {
            // TODO: em dashes I guess
            PdNum::Int(n)     => n.to_string(),
            PdNum::Float(f)   => f.to_string(),
            PdNum::Complex(z) => format!("{} {}j+", z.re, z.im),
            // TODO as above
            PdNum::Char(c)  => c.to_u32().and_then(std::char::from_u32).map_or_else(
                || ".'".to_string() + &c.to_string(),
                |ch| ['\'', ch].iter().collect::<String>())
        }
    }

    forward_int_coercion!(ceil);
    forward_int_coercion!(floor);
    forward_int_coercion!(trunc);
    forward_int_coercion!(round);

    pub fn abs(&self) -> PdNum {
        match self {
            PdNum::Int(k) => PdNum::Int(k.abs()),
            PdNum::Char(k) => PdNum::Char(k.abs()),
            PdNum::Float(f) => PdNum::Float(f.abs()),
            PdNum::Complex(z) => PdNum::Float(z.norm()),
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
                    self.clone()
                } else if *f == 0.0 {
                    PdNum::from(0)
                } else if *f > 0.0 {
                    PdNum::from(1)
                } else {
                    PdNum::from(-1)
                }
            }
            PdNum::Complex(z) => {
                if z.is_zero() {
                    self.clone()
                } else {
                    PdNum::Complex(z / z.norm())
                }
            }
        }
    }

    // because we want to preserve char-ness
    pub fn add_const(&self, k: i32) -> PdNum {
        match self {
            PdNum::Int(n) => PdNum::Int(n + k),
            PdNum::Char(c) => PdNum::Char(c + k),
            PdNum::Float(f) => PdNum::Float(f + (k as f64)),
            PdNum::Complex(z) => PdNum::Complex(z + (k as f64)),
        }
    }

    pub fn mul_const(&self, k: i32) -> PdNum {
        match self {
            PdNum::Int(n) => PdNum::Int(n * k),
            PdNum::Char(c) => PdNum::Char(c * k),
            PdNum::Float(f) => PdNum::Float(f * (k as f64)),
            PdNum::Complex(z) => PdNum::Complex(z * (k as f64)),
        }
    }

    pub fn mul_div_const(&self, mul: i32, div: i32) -> PdNum {
        self.mul_const(mul) / PdNum::from(div)
    }

    pub fn is_nonzero(&self) -> bool {
        match self {
            PdNum::Int(i) => !i.is_zero(),
            PdNum::Float(f) => *f != 0.0,
            PdNum::Char(c) => !c.is_zero(),
            PdNum::Complex(z) => !z.is_zero(),
        }
    }

    pub fn to_f64(&self) -> Option<f64> {
        match self {
            PdNum::Int(i) => i.to_f64(),
            PdNum::Float(f) => Some(*f),
            PdNum::Char(c) => c.to_f64(),
            PdNum::Complex(_) => None,
        }
    }

    pub fn to_f64_or_inf_or_complex(&self) -> Result<f64, Complex64> {
        match self {
            PdNum::Int(i) => Ok(bigint_to_f64_or_inf(i)),
            PdNum::Float(f) => Ok(*f),
            PdNum::Char(c) => Ok(bigint_to_f64_or_inf(c)),
            PdNum::Complex(z) => Err(*z),
        }
    }

    pub fn to_f64_re_or_inf(&self) -> f64 {
        match self.to_f64_or_inf_or_complex() {
            Err(z) => z.re,
            Ok(x) => x,
        }
    }

    pub fn to_complex_or_inf(&self) -> Complex64 {
        match self.to_f64_or_inf_or_complex() {
            Err(z) => z,
            Ok(x) => Complex64::from(x),
        }
    }

    pub fn is_pure_real(&self) -> bool {
        match self {
            PdNum::Int(_) => true,
            PdNum::Float(_) => true,
            PdNum::Char(_) => true,
            PdNum::Complex(z) => z.im == 0.0,
        }
    }

    pub fn is_pure_imaginary(&self) -> bool {
        match self {
            PdNum::Int(i) => i.is_zero(),
            PdNum::Float(f) => *f == 0.0,
            PdNum::Char(c) => c.is_zero(),
            PdNum::Complex(z) => z.re == 0.0,
        }
    }

    pub fn conjugate(&self) -> PdNum {
        match self {
            PdNum::Int(_) => self.clone(),
            PdNum::Float(_) => self.clone(),
            PdNum::Char(_) => self.clone(),
            PdNum::Complex(z) => PdNum::Complex(z.conj()),
        }
    }

    pub fn swap_real_and_imaginary(&self) -> PdNum {
        match self.to_f64_or_inf_or_complex() {
            Ok(f) => PdNum::Complex(Complex64::new(0.0, f)),
            Err(z) => PdNum::Complex(Complex64::new(z.im, z.re)),
        }
    }

    pub fn real_part(&self) -> PdNum {
        match self {
            PdNum::Int(_) => self.clone(),
            PdNum::Float(_) => self.clone(),
            PdNum::Char(_) => self.clone(),
            PdNum::Complex(z) => PdNum::Float(z.re),
        }
    }

    pub fn imaginary_part(&self) -> PdNum {
        match self {
            PdNum::Int(_) => PdNum::Int(BigInt::from(0)),
            PdNum::Float(_) => PdNum::Float(0.0),
            PdNum::Char(_) => PdNum::Char(BigInt::from(0)),
            PdNum::Complex(z) => PdNum::Float(z.im),
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
            PdNum::Complex(z) => PdNum::Complex(z.powi(e as i32)),
        }
    }

    pub fn pow_num(&self, other: &PdNum) -> PdNum {
        macro_rules! powi_or_powf {
            ($a:expr, $b:expr) => {
                match $b.to_i32() {
                    Some(ib) => PdNum::from($a.powi(ib)),
                    None => PdNum::from($a.powf(bigint_to_f64_or_inf($b))),
                }
            };
        }

        match (self, other) {
            (PdNum::Int   (a), PdNum::Int   (b)) => pow_big_ints(a, b),
            (PdNum::Int   (a), PdNum::Float (b)) => PdNum::from(a.to_f64().expect("pow pls").pow(b)),
            (PdNum::Int   (a), PdNum::Char  (b)) => pow_big_ints(a, b),
            (PdNum::Char  (a), PdNum::Int   (b)) => pow_big_ints(a, b),
            (PdNum::Char  (a), PdNum::Float (b)) => PdNum::from(a.to_f64().expect("pow pls").pow(b)),
            (PdNum::Char  (a), PdNum::Char  (b)) => pow_big_ints(a, b),

            (PdNum::Float (a),  PdNum::Float(b)) => PdNum::from(a.pow(b)),
            (PdNum::Complex(a), PdNum::Float(b)) => PdNum::from(a.powf(*b)),

            (PdNum::Float  (a), PdNum::Int (b)) => powi_or_powf!(a, b),
            (PdNum::Float  (a), PdNum::Char(b)) => powi_or_powf!(a, b),
            (PdNum::Complex(a), PdNum::Int (b)) => powi_or_powf!(a, b),
            (PdNum::Complex(a), PdNum::Char(b)) => powi_or_powf!(a, b),

            (a, PdNum::Complex(zb)) => PdNum::Complex(a.to_complex_or_inf().pow(zb)),
        }
    }

    pub fn factorial(&self) -> PdNum {
        match self {
            PdNum::Int(a) => PdNum::Int(factorial_big_int(a)),
            PdNum::Char(a) => PdNum::Char(factorial_big_int(a)),
            PdNum::Float(f) => PdNum::Float(gamma::gamma(f + 1.0)),
            PdNum::Complex(_) => {
                panic!("OK you should be able to compute the factorial of a complex number but it's hard");
            }
        }
    }

    pub fn is_nan(&self) -> bool {
        match self {
            PdNum::Int(_) => false,
            PdNum::Float(f) => f.is_nan(),
            PdNum::Char(_) => false,
            PdNum::Complex(z) => z.re.is_nan() || z.im.is_nan(),
        }
    }

    pub fn to_isize(&self) -> Option<isize> {
        match self {
            PdNum::Int(n) => n.to_isize(),
            PdNum::Float(f) => f.trunc().to_isize(),
            PdNum::Char(c) => c.to_isize(),
            PdNum::Complex(z) => z.re.trunc().to_isize(),
        }
    }

    pub fn to_usize(&self) -> Option<usize> {
        match self {
            PdNum::Int(n) => n.to_usize(),
            PdNum::Float(f) => f.trunc().to_usize(),
            PdNum::Char(c) => c.to_usize(),
            PdNum::Complex(z) => z.re.trunc().to_usize(),
        }
    }

    pub fn to_char(&self) -> Option<char> {
        std::char::from_u32(match self {
            PdNum::Int(n) => n.to_u32()?,
            PdNum::Float(f) => f.trunc().to_u32()?,
            PdNum::Char(c) => c.to_u32()?,
            PdNum::Complex(z) => z.re.trunc().to_u32()?,
        })
    }

    pub fn to_clamped_usize(&self) -> usize {
        match self {
            PdNum::Int(n) => {
                if !n.is_positive() { 0usize } else { n.to_usize().unwrap_or(usize::MAX) }
            }
            PdNum::Float(f) => {
                if *f <= 0.0 || f.is_nan() { 0usize } else { f.trunc().to_usize().unwrap_or(usize::MAX) }
            }
            PdNum::Char(c) => {
                if !c.is_positive() { 0usize } else { c.to_usize().unwrap_or(usize::MAX) }
            }
            PdNum::Complex(z) => {
                if z.re <= 0.0 || z.re.is_nan() { 0usize } else { z.re.trunc().to_usize().unwrap_or(usize::MAX) }
            }
        }
    }

    pub fn to_nn_usize(&self) -> Option<usize> {
        let s = self.to_usize()?;
        if s == 0 { None } else { Some(s) }
    }

    pub fn trunc_to_bigint(&self) -> Option<BigInt> {
        match self {
            PdNum::Int(n) => Some(BigInt::clone(n)),
            PdNum::Float(f) => f.trunc().to_bigint(),
            PdNum::Char(c) => Some(BigInt::clone(c)),
            PdNum::Complex(z) => z.re.trunc().to_bigint(),
        }
    }

    pub fn construct_like_self(&self, n: BigInt) -> PdNum {
        match self {
            PdNum::Int(_)   => PdNum::Int(n),
            PdNum::Float(_) => PdNum::Int(n),
            PdNum::Char(_)  => PdNum::Char(n),
            PdNum::Complex(_) => PdNum::Int(n),
        }
    }

    pub fn through_float<F>(&self, f: F) -> PdNum where F: FnOnce(f64) -> f64 {
        PdNum::Float(match self.to_f64() {
            Some(x) => f(x),
            None => f64::NAN,
        })
    }

    pub fn iverson(b: bool) -> Self { PdNum::from(if b { 1 } else { 0 }) }

}

// this seems... nontrivial??
fn cmp_bigint_f64(a: &BigInt, b: &f64) -> Option<Ordering> {
    if let Some(bi) = b.to_bigint() {
        Some(a.cmp(&bi))
    } else if b.is_infinite() {
        if b.is_sign_positive() {
            Some(Ordering::Less)
        } else {
            Some(Ordering::Greater)
        }
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

// useful to project down to this for ease of doing stuff
enum PdNumReal<'a> {
    Int(&'a BigInt),
    Float(f64),
}

impl<'a> PartialEq for PdNumReal<'a> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PdNumReal::Int  (a), PdNumReal::Int  (b)) => a == b,
            (PdNumReal::Int  (a), PdNumReal::Float(b)) => b.to_bigint().map_or(false, |x| &x == *a),
            (PdNumReal::Float(a), PdNumReal::Int  (b)) => a.to_bigint().map_or(false, |x| &x == *b),
            (PdNumReal::Float(a), PdNumReal::Float(b)) => a == b,
        }
    }
}

impl<'a> PartialOrd for PdNumReal<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (PdNumReal::Int  (a), PdNumReal::Int  (b)) => Some(a.cmp(b)),
            (PdNumReal::Int  (a), PdNumReal::Float(b)) => cmp_bigint_f64(a, b),
            (PdNumReal::Float(a), PdNumReal::Int  (b)) => cmp_bigint_f64(b, a).map(|ord| ord.reverse()),
            (PdNumReal::Float(a), PdNumReal::Float(b)) => a.partial_cmp(b),
        }
    }
}

impl<'a> PdNumReal<'a> {
    fn total_cmp_small_nan(&self, other: &Self) -> Ordering {
        match (self, other) {
            (PdNumReal::Int  (a), PdNumReal::Int  (b)) => a.cmp(b),
            (PdNumReal::Int  (a), PdNumReal::Float(b)) => cmp_bigint_f64(a, b).unwrap_or(Ordering::Greater),
            (PdNumReal::Float(a), PdNumReal::Int  (b)) => cmp_bigint_f64(b, a).map_or(Ordering::Less, |ord| ord.reverse()),
            (PdNumReal::Float(a), PdNumReal::Float(b)) => a.partial_cmp(b).unwrap_or(b.is_nan().cmp(&a.is_nan())), // note swap
        }
    }

    fn total_cmp_big_nan(&self, other: &Self) -> Ordering {
        match (self, other) {
            (PdNumReal::Int  (a), PdNumReal::Int  (b)) => a.cmp(b),
            (PdNumReal::Int  (a), PdNumReal::Float(b)) => cmp_bigint_f64(a, b).unwrap_or(Ordering::Less),
            (PdNumReal::Float(a), PdNumReal::Int  (b)) => cmp_bigint_f64(b, a).map_or(Ordering::Greater, |ord| ord.reverse()),
            (PdNumReal::Float(a), PdNumReal::Float(b)) => a.partial_cmp(b).unwrap_or(a.is_nan().cmp(&b.is_nan())),
        }
    }
}

impl<'a> PdNum {
    fn project_to_reals(&'a self) -> (PdNumReal<'a>, PdNumReal<'a>) {
        match self {
            PdNum::Int(a) => (PdNumReal::Int(a), PdNumReal::Float(0.0)),
            PdNum::Float(a) => (PdNumReal::Float(*a), PdNumReal::Float(0.0)),
            PdNum::Char(a) => (PdNumReal::Int(a), PdNumReal::Float(0.0)),
            PdNum::Complex(z) => (PdNumReal::Float(z.re), PdNumReal::Float(z.im)),
        }
    }
}

impl PartialEq for PdNum {
    fn eq(&self, other: &Self) -> bool {
        self.project_to_reals() == other.project_to_reals()
    }
}

// TODO: Watch https://github.com/rust-lang/rust/issues/72599, we will probably want total
// orderings in some cases.

impl PartialOrd for PdNum {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.project_to_reals().partial_cmp(&other.project_to_reals())
    }
}

impl PdNum {
    // (considers NaNs equal)
    fn total_cmp_small_nan(&self, other: &Self) -> Ordering {
        let (ra, ia) = self.project_to_reals();
        let (rb, ib) = other.project_to_reals();
        ra.total_cmp_small_nan(&rb).then(ia.total_cmp_small_nan(&ib))
    }

    fn total_cmp_big_nan(&self, other: &Self) -> Ordering {
        let (ra, ia) = self.project_to_reals();
        let (rb, ib) = other.project_to_reals();
        ra.total_cmp_big_nan(&rb).then(ia.total_cmp_big_nan(&ib))
    }
}


// Tries to follow the laws
#[derive(Debug, Clone)]
pub struct PdTotalNum(pub Rc<PdNum>);

impl fmt::Display for PdTotalNum {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{}t", self.0)
    }
}


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

fn consistent_hash_f64<H: Hasher>(f: f64, state: &mut H) {
    match f.to_bigint() {
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

impl Hash for PdTotalNum {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match &***self {
            PdNum::Int(a) => BigInt::hash(a, state),
            PdNum::Char(c) => BigInt::hash(c, state),
            PdNum::Float(f) => consistent_hash_f64(*f, state),
            PdNum::Complex(z) => {
                consistent_hash_f64(z.re, state);
                consistent_hash_f64(z.im, state);
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

    pub fn min_consuming(self, other: Self) -> PdNum {
        match self.total_cmp_big_nan(&other) {
            Ordering::Greater => other,
            _ => self,
        }
    }

    pub fn max_consuming(self, other: Self) -> PdNum {
        match self.total_cmp_small_nan(&other) {
            Ordering::Greater => self,
            _ => other,
        }
    }
}

// ????????
macro_rules! binary_match {
    ($a:expr, $b:expr, $method:ident, $intmethod:expr, $floatmethod:expr, $complexmethod:expr) => {
        match ($a, $b) {
            (PdNum::Complex(za), b) => PdNum::Complex($complexmethod(*za, b.to_complex_or_inf())),
            (a, PdNum::Complex(zb)) => PdNum::Complex($complexmethod(a.to_complex_or_inf(), *zb)),

            (PdNum::Float(fa), b) => PdNum::Float($floatmethod(*fa, b.to_f64_or_inf_or_complex().expect("complex not elim"))),
            (a, PdNum::Float(fb)) => PdNum::Float($floatmethod(a.to_f64_or_inf_or_complex().expect("complex not elim"), *fb)),

            (PdNum::Int  (a), PdNum::Int  (b)) => PdNum::Int($intmethod(a, b)),
            (PdNum::Int  (a), PdNum::Char (b)) => PdNum::Int($intmethod(a, b)),
            (PdNum::Char (a), PdNum::Int  (b)) => PdNum::Int($intmethod(a, b)),
            (PdNum::Char (a), PdNum::Char (b)) => PdNum::Char($intmethod(a, b)),
        }
    };
}

macro_rules! def_binary_method {
    ($method:ident, $intmethod:expr, $floatmethod:expr, $complexmethod:expr) => {
        fn $method(self, other: &PdNum) -> PdNum {
            binary_match!(self, other, $method, $intmethod, $floatmethod, $complexmethod)
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
    ($imp:ident, $method:ident, $intmethod:expr, $floatmethod:expr, $complexmethod:expr) => {
        impl $imp<&PdNum> for &PdNum {
            type Output = PdNum;

            def_binary_method!($method, $intmethod, $floatmethod, $complexmethod);
        }

        forward_impl_binary_method!($imp, $method);
    };
}

fn dumb_complex_div_floor(a: Complex64, b: Complex64) -> Complex64 {
    let c = a / b;
    Complex64::new(c.re.floor(), c.im.floor())
}

impl PdNum {
    pub fn div_floor(&self, other: &PdNum) -> PdNum {
        binary_match!(self, other, div_floor, Integer::div_floor, f64::div_euclid, dumb_complex_div_floor)
    }
}

impl_binary_method!(Add, add, Add::add, Add::add, Add::add);
impl_binary_method!(Sub, sub, Sub::sub, Sub::sub, Sub::sub);
impl_binary_method!(Mul, mul, Mul::mul, Mul::mul, Mul::mul);
impl_binary_method!(Rem, rem, Integer::mod_floor, f64::rem_euclid, Rem::rem);

impl Div<&PdNum> for &PdNum {
    type Output = PdNum;
    fn div(self, other: &PdNum) -> PdNum {
        let a = self.to_f64_or_inf_or_complex();
        let b = other.to_f64_or_inf_or_complex();
        match (a, b) {
            (Ok (fa), Ok (fb)) => PdNum::Float(fa / fb),
            (Err(za), Ok (fb)) => PdNum::Complex(Complex64::from(za / fb)),
            (Ok (fa), Err(zb)) => PdNum::Complex(Complex64::from(fa / zb)),
            (Err(za), Err(zb)) => PdNum::Complex(Complex64::from(za / zb)),
        }
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
            PdNum::Complex(z) => PdNum::Complex(-z),
        }
    }
}

impl AddAssign<&PdNum> for PdNum {
    fn add_assign(&mut self, other: &PdNum) {
        let n = mem::replace(self, PdNum::from(0));
        *self = &n + other;
    }
}

impl Sum for PdNum {
    fn sum<I: Iterator<Item=Self>>(iter: I) -> Self {
        iter.fold(PdNum::from(0), Add::add)
    }
}

impl Product for PdNum {
    fn product<I: Iterator<Item=Self>>(iter: I) -> Self {
        iter.fold(PdNum::from(1), Mul::mul)
    }
}

macro_rules! force_bi_binary_match {
    ($a:expr, $b:expr, $method:ident, $intmethod:expr) => {
        match ($a, $b) {
            (PdNum::Char(a), PdNum::Char(b)) => PdNum::Char($intmethod(a, b)),
            (na, nb) => match (na.trunc_to_bigint(), nb.trunc_to_bigint()) {
                (Some(ia), Some(ib)) => PdNum::Int($intmethod(&ia, &ib)),
                _ => PdNum::Float(f64::NAN),
            }
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

impl PdNum {
    pub fn gcd(&self, other: &PdNum) -> PdNum {
        force_bi_binary_match!(self, other, gcd, Integer::gcd)
    }

    pub fn shl_opt(&self, other: &PdNum) -> Option<PdNum> {
        let shift_amount: usize = other.to_usize()?;
        Some(match self {
            PdNum::Int(a) => PdNum::Int(a << shift_amount),
            PdNum::Char(a) => PdNum::Char(a << shift_amount),
            PdNum::Float(a) => PdNum::Int(a.trunc().to_bigint()? << shift_amount),
            PdNum::Complex(a) => PdNum::Int(a.re.trunc().to_bigint()? << shift_amount),
        })
    }

    pub fn shr_opt(&self, other: &PdNum) -> Option<PdNum> {
        let shift_amount: usize = other.to_usize()?;
        Some(match self {
            PdNum::Int(a) => PdNum::Int(a >> shift_amount),
            PdNum::Char(a) => PdNum::Char(a >> shift_amount),
            PdNum::Float(a) => PdNum::Int(a.trunc().to_bigint()? >> shift_amount),
            PdNum::Complex(a) => PdNum::Int(a.re.trunc().to_bigint()? >> shift_amount),
        })
    }
}
