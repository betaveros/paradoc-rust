use num::bigint::BigInt;

pub fn to_base_digits(base: &BigInt, num0: BigInt) -> Vec<BigInt> {
    if num0 == BigInt::from(0) {
        vec![BigInt::from(0)]
    } else {
        let (sign, mut num) = if num0 < BigInt::from(0) {
            (-1, -num0)
        } else {
            (1, num0)
        };
        let mut acc = Vec::new();

        while &num > &BigInt::from(0) {
            acc.push(&num % base);
            num /= base;
        }

        acc.iter().rev().map(|d| d * sign).collect()
    }
}

pub fn from_base_digits(base: &BigInt, digits: &[BigInt]) -> BigInt {
    let mut acc = BigInt::from(0);
    for digit in digits {
        acc = base * acc + digit
    }
    acc
}
