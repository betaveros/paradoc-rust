use std::io;
use std::io::{BufRead};
use std::fmt::Debug;

use num::bigint::BigInt;

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum InputTrigger {
    All,
    Line,
    Word,
    Value,
    Record, // Value but for a line
    // Char, // https://github.com/rust-lang/rust/issues/27802#issuecomment-377537778
    // This seems to actually be incredibly difficult to implement unless we handroll buffering
    // or something.
    AllLines,
    AllValues,
    AllRecords,
}

#[derive(Debug)]
pub enum ReadValue {
    String(String),
    Char(char),
    Int(BigInt),
    Float(f64),
    List(Vec<ReadValue>),
}

fn parse_read_value(s: String) -> ReadValue {
    match s.parse::<BigInt>() {
        Ok(n) => ReadValue::Int(n),
        _ => match s.parse::<f64>() {
            Ok(f) => ReadValue::Float(f),
            _ => ReadValue::String(s),
        }
    }
}

// we need our own eof state because if we're asked to read all of input and input is empty, we
// want to return "" exactly once and None thereafter.
pub struct EOFReader {
    reader: Box<dyn BufRead>,
    eof: bool,
}

impl EOFReader {
    pub fn new(reader: Box<dyn BufRead>) -> EOFReader {
        EOFReader { reader, eof: false }
    }
}

impl Debug for EOFReader {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(fmt, "EOFReader {{ eof: {:?}, reader: ??? }}", self.eof)
    }
}

fn is_u8_space(x: u8) -> bool { x == 9u8 || x == 10u8 || x == 13u8 || x == 32u8 }

impl EOFReader {
    fn read_line(&mut self) -> io::Result<Option<String>> {
        if self.eof { panic!("EOFRead internal shouldn't have called read_line after eof"); }

        let mut s = String::new();
        let read_count = self.reader.read_line(&mut s)?;
        if read_count == 0 {
            self.eof = true; Ok(None)
        } else {
            if s.ends_with('\n') { s.pop(); }
            if s.ends_with('\r') { s.pop(); }
            Ok(Some(s))
        }
    }

    fn read_u8(&mut self) -> io::Result<Option<u8>> {
        if self.eof { panic!("EOFRead internal shouldn't have called read_char after eof"); }

        let mut buffer = [0];
        let read_count = self.reader.read(&mut buffer)?;
        if read_count == 0 {
            self.eof = true; Ok(None)
        } else {
            Ok(Some(buffer[0]))
        }
    }

    fn read_word(&mut self) -> io::Result<Option<String>> {
        if self.eof { panic!("EOFRead internal shouldn't have called read_word after eof"); }

        let mut vec = Vec::new();
        let mut bb: Option<u8> = self.read_u8()?;
        while bb.map_or(false, is_u8_space) {
            bb = self.read_u8()?;
        }
        if bb.is_none() { return Ok(None) }

        while let Some(b) = bb.filter(|b| !is_u8_space(*b)) {
            vec.push(b);
            bb = self.read_u8()?;
        }
        if bb.is_none() { self.eof = true; }

        Ok(Some(std::str::from_utf8(&vec).expect("utf-8 fail in input").to_string()))
    }

    fn read_value(&mut self) -> io::Result<Option<ReadValue>> {
        let s = self.read_word()?;

        Ok(s.map(parse_read_value))
    }

    fn read_record(&mut self) -> io::Result<Option<ReadValue>> {
        let s = self.read_line()?;

        Ok(s.map(parse_read_value))
    }

    pub fn read(&mut self, trigger: InputTrigger) -> io::Result<Option<ReadValue>> {
        if self.eof { return Ok(None) }

        match trigger {
            InputTrigger::All => {
                let mut s = String::new();
                self.reader.read_to_string(&mut s)?;
                self.eof = true;
                Ok(Some(ReadValue::String(s)))
            }
            InputTrigger::Line => {
                Ok(self.read_line()?.map(|s| ReadValue::String(s)))
            }
            InputTrigger::Word => {
                Ok(self.read_word()?.map(|s| ReadValue::String(s)))
            }
            InputTrigger::Value => self.read_value(),
            InputTrigger::Record => self.read_record(),

            InputTrigger::AllLines => {
                let mut vec = Vec::new();
                while let Some(s) = self.read_line()? {
                    vec.push(ReadValue::String(s));
                }
                Ok(Some(ReadValue::List(vec)))
            }
            InputTrigger::AllValues => {
                let mut vec = Vec::new();
                while let Some(v) = self.read_value()? {
                    vec.push(v);
                }
                Ok(Some(ReadValue::List(vec)))
            }
            InputTrigger::AllRecords => {
                let mut vec = Vec::new();
                while let Some(v) = self.read_record()? {
                    vec.push(v);
                }
                Ok(Some(ReadValue::List(vec)))
            }
        }
    }
}
