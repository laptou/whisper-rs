#[cfg(test)]
pub mod test;

#[allow(unused_macros)]
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

#[allow(unused_imports)]
pub(crate) use tensor_dbg;
