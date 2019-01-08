import numpy as np
import torch


def get_all_formats(matrix):
    """Converts a scipy sparse matrix to a list of copies in formats that should
    be supported

    :param matrix: labeling function output matrix in a scipy sparse format
    :return: list of labeling function output matrices
    """
    other_formats = [
        matrix.todense(),
        matrix.todense().tolist(),
        matrix.todense().astype(np.float),
        matrix.tocoo(),
        matrix.tocsc(),
        matrix.todia(),
        matrix.todok(),
        matrix.tolil(),
        torch.tensor(matrix.todense())
    ]
    return other_formats
