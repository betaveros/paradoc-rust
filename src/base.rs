use num::bigint::{BigInt, Sign};
use num_traits::cast::ToPrimitive;
use num_traits::sign::Signed;
use num_traits::identities::{Zero, One};
use num::Integer;

use crate::vec_util::replicate_clones;
use crate::string_util::flat_collect_strs;

pub fn to_base_digits(base: &BigInt, num0: BigInt) -> Option<Vec<BigInt>> {
    if base >= &BigInt::from(2) {
        if num0.is_zero() {
            Some(vec![BigInt::from(0)])
        } else {
            let (sign, mut num) = if num0.is_negative() {
                (-1, -num0)
            } else {
                (1, num0)
            };
            let mut acc = Vec::new();

            while num.is_positive() {
                acc.push(&num % base);
                num /= base;
            }

            Some(acc.iter().rev().map(|d| d * sign).collect())
        }
    } else if base.is_one() {
        let length = num0.magnitude().to_usize().unwrap_or(usize::MAX);
        let fill = match num0.sign() {
            Sign::Plus => BigInt::from(1),
            _ => BigInt::from(-1),
        };
        Some(replicate_clones(length, &fill))
    } else if base <= &BigInt::from(-2) {
        let pos_base = -base;
        if num0.is_zero() {
            Some(vec![BigInt::from(0)])
        } else {
            let mut num = num0;
            let mut acc = Vec::new();

            while !num.is_zero() {
                acc.push(num.mod_floor(&pos_base));
                num = -num.div_floor(&pos_base);
            }

            acc.reverse();
            Some(acc)
        }
    } else {
        None
    }
}

pub fn from_base_digits(base: &BigInt, digits: &[BigInt]) -> BigInt {
    let mut acc = BigInt::from(0);
    for digit in digits {
        acc = base * acc + digit
    }
    acc
}

const DIGITS_LOWER: [char; 36] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'];
const DIGITS_UPPER: [char; 36] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'];

fn to_base_digits_with_alphabet<S>(base: &BigInt, num0: BigInt, alphabet: [char; 36], internal_prepend_negative: bool) -> Option<S> where S: Default + Extend<char> {
    if base.is_positive() && num0.is_negative() {
        to_base_digits_with_alphabet(base, -num0, alphabet, true)
    } else {
        let mut neg_left = internal_prepend_negative;
        // this just returns Some("-") and then None, or None
        let neg_iter = std::iter::from_fn(move || {
            if neg_left {
                neg_left = false;
                Some("-".to_string())
            } else {
                None
            }
        });
        to_base_digits(base, num0).map(|digs| {
            flat_collect_strs(neg_iter.chain(digs.iter().map(|dig: &BigInt| -> String {
                match dig.to_usize().and_then(|digu| alphabet.get(digu)) {
                    Some(c) => c.to_string(),
                    None => format!("({})", dig),
                }
            })))
        })
    }
}

pub fn to_base_digits_lower<S>(base: &BigInt, num0: BigInt) -> Option<S> where S: Default + Extend<char> {
    to_base_digits_with_alphabet(base, num0, DIGITS_LOWER, false)
}
pub fn to_base_digits_upper<S>(base: &BigInt, num0: BigInt) -> Option<S> where S: Default + Extend<char> {
    to_base_digits_with_alphabet(base, num0, DIGITS_UPPER, false)
}

pub fn from_char_digit(ch0: &BigInt) -> Option<BigInt> {
    let chu = ch0.to_u32()?;
    let ch = std::char::from_u32(chu)?;
    if '0' <= ch && ch <= '9' {
        Some(BigInt::from(chu - ('0' as u32)))
    } else if 'A' <= ch && ch <= 'Z' {
        Some(BigInt::from(chu - ('A' as u32) + 10))
    } else if 'a' <= ch && ch <= 'z' {
        Some(BigInt::from(chu - ('a' as u32) + 10))
    } else {
        None
    }
}
