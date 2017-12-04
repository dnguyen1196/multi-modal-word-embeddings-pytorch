NCELoss inherits from nn.Module
- it inherits a function __call__(*input, **kwargs) which seems to get called
in main
- This function --__call__-- then in turn calls the self.forward function which is
implemented in NCELoss (or any child class of nn.Module)
- The forward function calculates the value of the input transformation to 
output (using neural network structure). It then calls the loss function which
basically calls the __criterion__ function.
- The criterion is actually a different class which also inherits from __nn.Module__
This class will be evoked with function __call(input, args, etc)__ which in turn
calls its __forward__ function (we implement)
- The __criterion.forward__ function then calls __get_probs__ function which simply
computes the cross product of the word embeddings. In the get_probs function, 
the main function that does the cross product is calling __self.encoder__
- The variable __self.encoder__ inherits class __nn.Linear__ (in fact, it can be
found in the same file down the NCELoss class). It also contains the __forward__
function which is used to perform matrix multiplication.

### TODO: Look at the implementation of LinearIndexer.forward


