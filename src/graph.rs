// Module for the Graph struct, which is useful later on for the recommendation system.

type AdjList = Vec<Vec<usize>>;

/// Simple struct for storing adjacency lists and reading data from them.
pub struct Graph {
    pub adjacency: AdjList,
}

impl Graph {
    /// Creates a new Graph object from a given adjacency list.
    /// Input: adjacency list.
    /// Output: a new Graph object.
    pub fn new(adjacency: AdjList) -> Self {
        Graph {adjacency}
    }

    /// Gets all the neighbors of a specified node within the Graph.
    /// Input: a specified node number.
    /// Output: all the neighbors of the given node.
    pub fn neighbors(&self, node: usize) -> Option<&Vec<usize>> {
        self.adjacency.get(node)
    }
}

#[cfg(test)]
mod tests {
    use super::Graph;

    #[test]
    fn test_neighbors() {
        let adj = vec![vec![1, 2], vec![0], vec![0]];
        let g = Graph::new(adj);
        assert_eq!(g.neighbors(0), Some(&vec![1, 2]));
    }
}