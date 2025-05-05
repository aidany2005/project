mod preprocessing;
mod graph;

use preprocessing::{load_and_preprocess, build_graph_from_features, load_metadata};
use graph::Graph;
use std::io;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "fashion_products.csv";

    // This can be changed depending on your preference
    let k_neighbors = 5;

    println!("Provide the Product ID for the product you would like recommendations for:");

    // Gets the user input and converts to node number
    let mut input = String::new(); 
    io::stdin().read_line(&mut input).expect("Failed to read line"); 
    let input = input.trim(); 

    let mut target_node: usize = input.parse()?;
    target_node -= 1;

    // Parameters for recommendations. Can also be changed based on preference
    let numeric = ["Price", "Rating"];
    let categorical = ["Brand", "Size"];

    let items = load_metadata(path)?;

    // Sorting, encoding data and building the recommendations
    let features = load_and_preprocess(path, &numeric, &categorical)?;
    let adjacency = build_graph_from_features(&features, k_neighbors);
    let graph = Graph::new(adjacency);

    let recs = recommend(&graph, target_node, k_neighbors);

    println!("Recommendations for {} from {}:", items[target_node].name, items[target_node].brand);

    for idx in recs {
        let item = &items[idx];
        println!("Product ID: {:>3}, Name: {:>7}, Brand: {}, Category: {:>15}, Price: {:>3}, Rating: {:.3}, Color: {:>6}, Size: {}", item.product, item.name, item.brand, item.category, item.price, item.rating, item.color, item.size);
    }

    Ok(())
}

/// Takes the top k neighbors of a given node and turns into a Vector.
/// Inputs: a Graph object, a node number and a value k
/// Output: a Vector of usizes, which is all node numbers.
fn recommend(graph: &Graph, node: usize, k: usize) -> Vec<usize> {
    graph.neighbors(node).map(|neigh| neigh.iter().cloned().take(k).collect()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{recommend};
    use crate::graph::Graph;

    #[test]
    fn test_recommend() {
        let adj = vec![vec![1, 2], vec![0], vec![0]];
        let g = Graph::new(adj);
        assert_eq!(recommend(&g, 0, 1), vec![1]);
        assert_eq!(recommend(&g, 1, 5), vec![0]);
    }
}