# Scenario: You have a new class of PDE and want to see if PDE surrogates can be useful for you

There are hundreds of possible PDE surrogate models. As with most deep learning, choice of hyperparameters can matter a lot too.
You might also have your own runtime requirements, and data or time limitations.
PDEArena provides a _simple interface_ for you to try out many possible model designs and understand the best model for your task and constraints.
All you need to do is write a `IterDataPipe` for your dataset, and you should be good to go.
