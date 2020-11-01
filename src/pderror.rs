#[derive(Debug)]
pub enum PdError {
    EmptyStack(String),
    UndefinedVariable(String),
    InapplicableTrailer(String),
    BadArgument(String),
    BadList(&'static str),
    NumericError(&'static str),
    UnhashableBlock(String),
    IndexError(String),
    EmptyResult(String),
    AssertError(String),
    BadAssignment(String),
    InputError,
    BadComparison,
    BadParse,
    BadFloat,
    EmptyReduceIntermediate,
    InvalidHoardOperation,
    Break,
    Continue,
}

pub type PdResult<T> = Result<T, PdError>;
pub type PdUnit = PdResult<()>;
