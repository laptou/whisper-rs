use std::path::Path;

use anyhow::Context;
use tch::Tensor;

pub fn read_csv_2d(path: impl AsRef<Path>) -> anyhow::Result<Tensor>
{
    let vals: Result<Vec<Vec<_>>, _> = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .context("failed to read csv file")?
        .into_records()
        .map(|r| {
            r.context("failed to read row from csv file")?
                .into_iter()
                .map(|s| {
                    s.parse::<f32>()
                        .context("failed to parse element from csv file")
                })
                .collect()
        })
        .collect();
    let vals = vals.unwrap();

    Ok(Tensor::of_slice2(&vals))
}
