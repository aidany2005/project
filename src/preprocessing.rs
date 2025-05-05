// Module for processing and sorting data into the necessary data structures for my recommendation system.

use std::error::Error;
use std::collections::{HashMap, HashSet};
use csv::Reader;
use ndarray::Array2;

/// Passes through the data twice, sorting data into columns then normalizing/encoding into one-hot vectors after.
/// Inputs: a file path, numeric column labels and categorical column labels.
/// Outputs: a feature matrix to be used later on.
/// The first loop through records is simply for storing data as minimums/maximums and categories.
/// The second loop then normalizes data using these mins/maxes and puts categorical labels into one-hot vectors.
/// All the data is put into one matrix and then outputted.
pub fn load_and_preprocess(path: &str, numeric_cols: &[&str], categorical_cols: &[&str],
) -> Result<Array2<f64>, Box<dyn Error>> {
    // First pass: stats and row count
    let mut rdr1 = Reader::from_path(path)?;

    // Takes all the headers and puts them into one map for easy access
    let headers = rdr1.headers()?.clone();
    let header_map: HashMap<_, _> = headers.iter().enumerate().map(|(i, h)| (h.to_string(), i)).collect();

    let mut categorical_values: HashMap<String, HashSet<String>> = HashMap::new();

    let mut mins = vec![f64::INFINITY; numeric_cols.len()];
    let mut maxs = vec![f64::NEG_INFINITY; numeric_cols.len()];

    let mut row_count = 0;

    for result in rdr1.records() {
        let rec = result?;
        row_count += 1;
        // Numeric stats
        for (i, &col) in numeric_cols.iter().enumerate() {
            let index = header_map[col];
            let v: f64 = rec[index].parse()?;
            if v < mins[i] {mins[i] = v;}
            if v > maxs[i] {maxs[i] = v;}
        }
        // Collect categories
        for &col in categorical_cols {
            let index = header_map[col];
            categorical_values.entry(col.to_string())
                .or_insert_with(HashSet::new)
                .insert(rec[index].to_string());
        }
    }

    // Sort category vectors
    let mut category_vecs: HashMap<String, Vec<String>> = HashMap::new();

    for (col, set) in categorical_values {
        let mut vec: Vec<_> = set.into_iter().collect();
        vec.sort();
        category_vecs.insert(col, vec);
    }

    // Determine feature length
    let mut feature_len = numeric_cols.len();

    for &col in categorical_cols {
        feature_len += category_vecs[col].len();
    }

    // Allocate matrix
    let mut mat = Array2::<f64>::zeros((row_count, feature_len));

    // Second pass: populate matrix
    let mut rdr2 = Reader::from_path(path)?;

    for (i, result) in rdr2.records().enumerate() {
        let rec = result?;
        let mut offset = 0;
        // Scaled numerics
        for (j, &col) in numeric_cols.iter().enumerate() {
            let index = header_map[col];
            let v: f64 = rec[index].parse()?;

            let scaled = (v - mins[j]) / (maxs[j] - mins[j]);
            mat[[i, offset]] = scaled;
            offset += 1;
        }
        // One-hot categories
        for &col in categorical_cols {
            let index = header_map[col];

            for (k, cat) in category_vecs[col].iter().enumerate() {
                mat[[i, offset + k]] = if rec[index] == *cat { 1.0 } else { 0.0 };
            }

            offset += category_vecs[col].len();
        }
    }

    Ok(mat)
}

/// Stores the different labels of each product as fields, useful later for printing outputs.
pub struct Item {
    pub product: i64,
    pub name: String,
    pub brand: String,
    pub category: String,
    pub price: i64,
    pub rating: f64,
    pub color: String,
    pub size: String,
}

/// Loads product metadata from CSV into items, which are then placed into a Vector of items.
/// Input: takes a file path.
/// Output: returns a Vector of items or an error.
pub fn load_metadata(path: &str) -> Result<Vec<Item>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut items = Vec::new();

    // Takes data from each node and places into an Item. Relatively crude method but it works regardless
    for result in rdr.records() {
        let rec = result?;

        let product: i64 = rec[1].parse()?;
        let name: String = rec[2].parse()?;
        let brand: String = rec[3].parse()?;
        let category: String = rec[4].parse()?;
        let price: i64 = rec[5].parse()?;
        let rating: f64 = rec[6].parse()?;
        let color: String = rec[7].parse()?;
        let size: String = rec[8].parse()?;

        items.push(Item {product, name, brand, category, price, rating, color, size});
    }

    Ok(items)
}

/// Takes the previously created feature matrix and converts into an adjacency list.
/// Inputs: a feature matrix and a value k.
/// Output: a Vector of Vectors of usizes.
/// Loops through all of the features and calculates cosine similarity to every other node.
pub fn build_graph_from_features(features: &Array2<f64>, k: usize) -> Vec<Vec<usize>> {
    let n = features.nrows();
    let mut graph = vec![Vec::new(); n];

    // Gets initial norms of each row of the feature matrix
    let norms: Vec<f64> = (0..n).map(|i| features.row(i).dot(&features.row(i)).sqrt()).collect();

    for i in 0..n {
        let row_i = features.row(i);

        // Calcualtes cosine similarity to all other nodes in the feature matrix
        let mut sim_list: Vec<(usize, f64)> = (0..n).filter(|&j| j != i).map(|j| {
            let dot = row_i.dot(&features.row(j));
            (j, dot / (norms[i] * norms[j]))
        }).collect();

        // Sorts nodes in descending order of highest similarity
        sim_list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        graph[i] = sim_list.into_iter().take(k).map(|(j, _)| j).collect();
    }

    graph
}

#[cfg(test)]
mod tests {
    use super::{load_and_preprocess, build_graph_from_features};
    use ndarray::{array, Array2};

    #[test]
    fn test_preprocess() {
        let data = "price,quality,brand,size
1.0,10.0,A,Small
2.0,20.0,B,Medium
";
        std::fs::write("test.csv", data).unwrap();
        let mat = load_and_preprocess("test.csv", &["price","quality"], &["brand","size"]).unwrap();
        assert_eq!(mat.nrows(), 2);
        assert_eq!(mat.ncols(), 6);
    }

    #[test]
    fn test_build_graph() {
        let features: Array2<f64> = array![
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ];
        let graph = build_graph_from_features(&features, 1);
        assert_eq!(graph[0][0], 1);
        assert_eq!(graph[1][0], 0);
        let neighbor_of_2 = graph[2][0];
        assert!(neighbor_of_2 == 0 || neighbor_of_2 == 1);
    }
}