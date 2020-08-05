#[derive(Debug)]
pub enum PdError {
    EmptyStack(String),
    UndefinedVariable,
    InapplicableTrailer,
    BadArgument(String),
    BadList(&'static str),
    NumericError(&'static str),
}

pub type PdResult<T> = Result<T, PdError>;
pub type PdUnit = PdResult<()>;
