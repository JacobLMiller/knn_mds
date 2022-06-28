# L2G

Graph Embedding from local to global

```
usage: layout.py [-h] [--max_iter MAX_ITER]
                [--learning_rate LEARNING_RATE] [--output OUTPUT]
                input_graph

Read a graph, and produce an embedding with local-to-global (L2G) algorithm.

positional arguments:
  input_graph

optional arguments:
  -h, --help            show this help message and exit

  --opt_scale           Whether to optimizing the scale of the data.
                        Defaults to 0, which will use a heuristic.
                        A 1 will use the optimization.

  --k -k                The number of most-connected neighbors to find.

  --alpha  -a           Strength of the repulsive force.

  --epsilon -e          Threshold for convergence.

  --learning_rate LEARNING_RATE, -l LEARNING_RATE
                        If given a number, will use a fixed schedule. Will also accept
                        'fraction' 'sqrt' and 'convergent' schedules. Defaults to 'convergent'

  --max_iter            Maximum number of iterations.

  --output OUTPUT, -o OUTPUT
                        Save layout to the specified file.
```

# Example:
```
# Read the input graph (a 10x10 grid), and display the result

python3 layout.py graphs/10square.dot
```

# Dependencies

* `python3`
* [`numpy`](http://www.numpy.org/)
* [`matplotlib`](https://matplotlib.org/)
* [`graph-tool`](https://graph-tool.skewed.de/)
* [`numba`](http://deeplearning.net/software/theano/)
* [`scikit-learn`](http://scikit-learn.org/stable/)
