import torch
from torch import nn


class LevenshteinLoss(nn.Module):
    """
    The Levenshtein distance is the minimum number of single-character edits (insertions, deletions
    or substitutions) required to change one word/sequence into the other.
    The two sequences do not have to have the same input length.
    :return: Levenshtein Distance between seq1 and seq2
    """

    def __init__(self):
        super(LevenshteinLoss, self).__init__()

    def forward(self, seq1, seq2):
        size1 = seq1.size(dim=0)
        size2 = seq2.size(dim=0)
        matrix = nn.parameter.Parameter(torch.zeros((size1, size2), requires_grad=True))
        print("matrix init grad", matrix[0, 0].grad)
        # initialize
        with torch.no_grad():
            for i in range(size1):
                matrix[i, 0] = i
            for i in range(size2):
                matrix[0, i] = i
        # Fill the matrix via Dynamic Programming Approach in O(size1*size2) complexity
        for i in range(1, size1):
            for j in range(1, size2):
                if torch.eq(seq1[i], seq2[j]):
                    val1 = matrix[i - 1, j].clone()
                    val2 = matrix[i - 1, j - 1].clone()
                    val3 = matrix[i, j - 1].clone()
                    matrix[i, j] = torch.min(
                        torch.tensor([
                            val1 + 1,
                            val2,
                            val3 + 1
                        ])
                    )
                else:
                    val1 = matrix[i - 1, j].clone()
                    val2 = matrix[i - 1, j - 1].clone()
                    val3 = matrix[i, j - 1].clone()
                    matrix[i, j] = torch.min(
                        torch.tensor([
                            val1 + 1,
                            val2 + 1,
                            val3 + 1
                        ])
                    )
        # print(matrix)
        print("matrix grad", matrix[size1 - 1, size2 - 2].grad)
        loss = matrix[size1 - 1, size2 - 1]
        loss.retain_grad()
        return loss



if __name__ == "__main__":
    seq_a = torch.Tensor([1, 0, 0, 1])
    seq_b = torch.Tensor([0, 0, 0, 2])
    model = nn.Sequential(nn.Linear(10, 4))
    #print("model grad", list(model.parameters())[0].grad)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3)
    loss_fn = LevenshteinLoss()
    #print("params before", list(model.parameters()))
    a = list(model.parameters())[0].clone()
    input_ten = torch.randn(10)
    output_ten = model(input_ten)
    lev_loss = loss_fn(input_ten, output_ten)
    optimizer.zero_grad()
    lev_loss.backward()
    optimizer.step()
    #print("params after", list(model.parameters()))
    b = list(model.parameters())[0].clone()
    print("params equal", torch.equal(a.data, b.data))
    print(f"Levenshtein distance: {lev_loss}")
