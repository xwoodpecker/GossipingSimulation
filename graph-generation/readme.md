# Graph Generation CLI

This Python script is a command-line tool for generating and exporting various types of graphs. 
It provides flexibility in choosing the number of graphs, the type of graph to generate, and exporting options.

## Getting Started

To use this tool, you need to have Python installed on your system.

1. Clone the repository to your local machine:
   ```bash
   git clone <repository-url>
   ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the CLI
    ```bash
    python graph_generation_cli.py
   ```
Command-Line Options:

--count: Specifies the number of graphs to generate. 
You can enter a single number (N) or a combination of MxN, where M is the number of different parameters, 
and N is the number of graphs generated for each set of parameters.

--graph-type: Allows you to choose the type of graph to generate from a predefined list of options.

### Interactive Prompts:

The CLI provides prompts for user input to specify the count and graph type. 
It handles various input formats for count.
The CLI supports the generation of multiple graphs with different parameters. I
t iterates through the specified count and creates graphs accordingly.
Users have the option to provide parameters either one by one, interpolate them or let them be randomly distributed.
After generating the graphs, the CLI computes various graph properties and metrics for each graph, 
such as modularity, average degree, clustering, assortativity, and many others.
Users can choose to select a subset of generated graphs based on a specific metric value to approximate. 
The CLI then identifies the closest graphs to the desired metric value.
Users can choose to export the generated graphs as adjacency lists or as Kubernetes (k8s) graph resource YAML files. 
They can also specify options for naming and node value assignment.
Additionally, there's an option to export the graphs in Gephi format for visualization.

## Examples
Here are some examples of how to use the CLI:

Generate 10 random scale-free graphs:

```bash
python graph_generation_cli.py --count=10 --graph-type=scale-free
```

Generate 5 graphs with different parameters, each with 10 variations:

```bash
python graph_generation_cli.py --count=5x10 --graph-type=barabasi-albert
```

Generate a subset of graphs with values close to a target metric:
```bash
python graph_generation_cli.py --count=20/10x5 --graph-type=fixed-popularity
```

## Resources
The CustomResourceDefinition for the graphs can be found under [./k8s/graph-resource.yaml](./k8s/graph-resource.yaml).
The graph resources that were created for the executed simulations can be found under [./k8s/simulation-series](./k8s/simulation-series).
Other example files can be found under [./k8s/examples](./k8s/examples).

## Sources
The complex graph generation algorithm was taken from the following project ["Random Modular Network Generator"](https://github.com/bansallab/modular_graph_generator). 