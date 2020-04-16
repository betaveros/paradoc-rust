use regex::Regex;
use num::bigint::BigInt;

#[derive(Debug, PartialEq, Eq)]
pub struct Trailer(pub String);

#[derive(Debug, PartialEq)]
pub enum Leader {
    StringLit(String),
    IntLit(BigInt),
    CharLit(char),
    FloatLit(f64),
    Block(Box<Vec<Token>>),
    Var(String),
}

#[derive(Debug, PartialEq)]
pub struct Token(pub Leader, pub Vec<Trailer>);

pub fn lex(code: &str) -> Vec<(&str, &str)> {
    lazy_static! {
        // group 1, group 2
        static ref PD_TOKEN_PATTERN: Regex = Regex::new(r#"\.\.[^\n\r]*|("(?:\\"|\\\\|[^"])*"|'.|[0-9]+(?:\.[0-9]+)?(?:e[0-9]+)?|[^"'0-9a-z_])([a-z_]*)"#).unwrap();
    }

    let mut caps = PD_TOKEN_PATTERN.capture_locations();

    let mut i = 0;
    let mut go = true;
    let mut tokens = Vec::new();

    while go {
        match PD_TOKEN_PATTERN.captures_read_at(&mut caps, code, i) {
            Some(match_result) => {
                match (caps.get(1), caps.get(2)) {
                    (Some((a, b)), Some((c, d))) => {
                        tokens.push((&code[a..b], &code[c..d]));
                    }
                    _ => {
                        // comment. meh.
                    }
                }
                i = match_result.end();
            },
            None => {
                go = false;
            },
        }
    }
    assert!(i == code.len(), "didn't finish lexing!");

    tokens
}

pub fn parse_string_leader(s: &str) -> String {
    // maybe TODO: is pushing onto a String performant?
    let mut ret = String::new();
    let mut cit = s.chars();
    assert!(cit.next() == Some('"'), "string leader starts with double quote");
    loop {
        match cit.next() {
            Some('\\') => {
                match cit.next() {
                    Some('\\') => { ret.push('\\'); }
                    Some('"')  => { ret.push('"'); }
                    Some(c)  => { ret.push('\\'); ret.push(c); }
                    None => { panic!("string leader should end with double quote; got backslash and nothing"); }
                }
            }
            Some('"') => { break; }
            Some(c) => { ret.push(c); }
            None => { panic!("string leader should end with double quote; got nothing"); }
        }
    }
    ret.shrink_to_fit();
    ret
}

pub fn parse_trailer(s: &str) -> Vec<Trailer> {
    let mut trailers = Vec::new();
    let mut cit = s.chars();
    loop {
        match cit.next() {
            Some('_') => {
                loop {
                    // right here, we have already consumed a _
                    // (and we need it here because the _s mattter for variables
                    let mut cur_trailer = "_".to_string();
                    let mut consumed_next_underscore = false;
                    loop {
                        match cit.next() {
                            Some('_') => {
                                consumed_next_underscore = true;
                                break;
                            }
                            Some(c) => {
                                cur_trailer.push(c);
                            }
                            None => { break; }
                        }
                    }
                    trailers.push(Trailer(cur_trailer));
                    if !consumed_next_underscore {
                        break;
                    }
                }
                break;
            }
            Some(c) => {
                trailers.push(Trailer(c.to_string()));
            }
            None => { break; }
        }
    }
    trailers
}

pub fn parse_at(tokens: &Vec<(&str, &str)>, start: usize) -> (Vec<Token>, usize) {
    let mut ret = Vec::new();
    let mut cur = start;
    while cur < tokens.len() {
        // TODO: trailers
        let (leader, trailer0) = tokens[cur];
        let mut trailer = trailer0;
        let (ld, next) = match leader.chars().next() { // first utf-8 char
            Some('"') => {(
                Leader::StringLit(parse_string_leader(leader)),
                cur + 1
            )}
            Some('\'') => {(
                Leader::CharLit(leader.chars().nth(1).unwrap()),
                cur + 1
            )}
            Some('{') => {
                let (inner, ni) = parse_at(tokens, cur + 1);
                assert_eq!(tokens[ni].0, "}", "inner parse should stop at close brace");
                trailer = tokens[ni].1;
                (Leader::Block(Box::new(inner)), ni + 1)
            }
            Some('}') => { break; }
            None => { break; }
            _ => match leader.parse::<BigInt>() {
                Ok(x) => { (Leader::IntLit(x), cur + 1) }
                _ => {
                    match leader.parse::<f64>() {
                        Ok(x) => { (Leader::FloatLit(x), cur + 1) }
                        _ => (Leader::Var(leader.to_string()), cur + 1)
                    }
                }
            }
        };
        ret.push(Token(ld, parse_trailer(trailer)));
        cur = next
    }
    (ret, cur)
}

pub fn parse(code: &str) -> Vec<Token> {
    let lexed = lex(code);
    let (toks, end) = parse_at(&lexed, 0);
    assert!(end == lexed.len(), "didn't finish parsing");
    toks
}
