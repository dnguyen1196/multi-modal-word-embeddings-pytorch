import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logger.disabled = True

import torch
import torch.optim as optimization
import nce_loss.data as data
from nce_loss.model import RNNModel
from nce_loss.nce import NCELoss
from nce_loss.utils import process_data, build_unigram_noise

batch_size = 128
data_path = "/home/duc/Documents/projects/tiki_torch/data/penn"

corpus = data.Corpus(path=data_path, batch_size=batch_size,shuffle=True)
nhidden = 200
embedding_size = 100
nlayers = 1
ntokens = len(corpus.train.dataset.dictionary)
print('Vocabulary size is {}'.format(ntokens))

use_cuda = True

# Create nce_loss from noise, criterion ce
noise = build_unigram_noise(torch.FloatTensor(corpus.train.dataset.dictionary.idx2count))

A = torch.rand([embedding_size, embedding_size])
fmat = torch.add(A,A)/2

# If we need to add a third matrix into the system, this is where we need to have it
criterion_nce = NCELoss(
    ntokens=ntokens,
    nhidden=nhidden,
    noise=noise,
    normed_eval=True,  # evaluate PPL using normalized prob
    cuda=use_cuda,
    factor_matrix=fmat
)

# The optimizer requires some parameters to optimize, the learning rate and momentum
model = RNNModel(
    ntoken=ntokens,
    ninp=embedding_size,
    nhid=nhidden,
    nlayers=nlayers,
    criterion=criterion_nce
)
if use_cuda:
    model.cuda()

# Turn on training mode which enables dropout
def train(model, data_source, lr=0.05, weight_decay=1e-5, momentum=0.9):
    log_interval = 100
    params = model.parameters()
    model.train()

    sgd = optimization.SGD(
        params=params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    num_epochs = 1

    for epoch in range(1, num_epochs + 1): # Loop through the entire corpus
        total_loss = 0.0 # Sort of reset total loss after each epochs

        for num_batch, data_batch in enumerate(data_source): #
            # Why do we have to manually reset gradients
            sgd.zero_grad()

            # Parse the data_batch into data, target and length
            data, target, length = process_data(data_batch, cuda=use_cuda)

            # Build a model from data, target, length
            logger.debug(["experiment.train.data ", data.size()])
            logger.debug(["experiment.train.target ", target.size()])
            logger.debug(["experiment.train.length ", length.size()])

            # The loss function seems to
            loss = model(data, target, length) # What does this function do
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm(params, 0.25) # Clipping gradient, default is 0.25

            # Stochastic gradient descent applies the update to the variables
            sgd.step()

            # Why does the loss value is loss.data[0]
            cur_loss = loss.data[0]
            total_loss += cur_loss

            if num_batch % log_interval == 0 and num_batch > 0:
                print(num_batch, " -> ", cur_loss)


train(model, corpus.train, lr=0.01, weight_decay=1e-5, momentum=0.9)