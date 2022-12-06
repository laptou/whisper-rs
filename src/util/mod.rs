#[cfg(test)]
pub mod test;

macro_rules! tensor_dbg {
    ($tensor: expr) => {
        eprintln!(
            "[{}:{}] {} => {}",
            file!(),
            line!(),
            stringify!($tensor),
            $tensor
        );
    };
}

pub(crate) use tensor_dbg;
