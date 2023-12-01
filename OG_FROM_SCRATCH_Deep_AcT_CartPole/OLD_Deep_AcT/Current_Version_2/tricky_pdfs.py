import torch
import math
import torch.distributions.multivariate_normal as mvn
import numpy as np


def diag_mvg_pdf_torch(samples, means, diag_vars):
    """
    Compute the PDF of a diagonal multivariate Gaussian distribution using PyTorch.

    Parameters:
    - samples (torch.Tensor): The samples tensor (batch_size x num_features).
    - means (torch.Tensor): The means tensor (batch_size x num_features).
    - diag_vars (torch.Tensor): The diagonal elements of the covariance matrix (batch_size x num_features).

    Returns:
    - pdf (torch.Tensor): The average PDF value over the batch.
    """
    batch_size = len(samples)

    final_pdf = 0

    for i in range(batch_size):
        mvn_dist_i = mvn.MultivariateNormal(
            loc=means[i],
            covariance_matrix=torch.diag(diag_vars[i])
        )

        pdf_i = torch.exp(mvn_dist_i.log_prob(samples[i]))
        final_pdf += pdf_i

        print(f"\npdf_i: {pdf_i}")
        print(f"samples[i]: {samples[i]}")
        print(f"means[i]: {means[i]}")
        print(f"diag_vars[i]: {diag_vars[i]}\n")

    # Compute the average PDF over the batch
    pdf = final_pdf / batch_size

    return pdf


# def diag_mvg_pdf_from_scratch(samples, means, diag_vars):
#     """
#     Compute the PDF of a diagonal multivariate Gaussian distribution using PyTorch.

#     Parameters:
#     - samples (torch.Tensor): The samples tensor (batch_size x num_features).
#     - means (torch.Tensor): The means tensor (batch_size x num_features).
#     - diag_vars (torch.Tensor): The diagonal elements of the covariance matrix (batch_size x num_features).

#     Returns:
#     - pdf (torch.Tensor): The average PDF value over the batch.
#     """
#     batch_size = len(samples)

#     final_pdf = 0

#     for i in range(batch_size):

#         exponent = ((samples[i] - means[i]) ** 2)/(2 * diag_vars[i])
#         normalization_term = (2*np.pi*diag_vars[i]) ** (-1/2)
#         pdf_i_dims = normalization_term * torch.exp(-exponent)
#         pdf_i = torch.prod(pdf_i_dims)

#         print(f"\nexponent: {exponent}")
#         print(f"normalization_term: {normalization_term}")
#         print(f"pdf_i_dims: {pdf_i_dims}")
#         print(f"pdf_i: {pdf_i}")
#         print(f"samples[i]: {samples[i]}")
#         print(f"means[i]: {means[i]}")
#         print(f"diag_vars[i]: {diag_vars[i]}\n")

#     # Compute the average PDF over the batch
#     pdf = final_pdf / batch_size

#     return pdf

def diag_mvg_pdf_from_scratch(sample, mean, diag_var):
    """
    Compute the PDF of a diagonal multivariate Gaussian distribution using PyTorch.

    Parameters:
    - sample (torch.Tensor): The sample tensor (batch_size x num_features).
    - mean (torch.Tensor): The mean tensor (batch_size x num_features).
    - diag_var (torch.Tensor): The diagonal elements of the covariance matrix (batch_size x num_features).

    Returns:
    - pdf (torch.Tensor): The average PDF value over the batch.
    """
    batch_size = len(sample)

    final_pdf = []

    for i in range(batch_size):

        exponent = ((sample[i] - mean[i]) ** 2)/(2 * diag_var[i])
        normalization_term = (2*np.pi*diag_var[i]) ** (-1/2)
        pdf_i = normalization_term * torch.exp(-exponent)
        final_pdf.append(pdf_i)

        print(f"\nexponent: {exponent}")
        print(f"normalization_term: {normalization_term}")
        print(f"pdf_i: {pdf_i}")
        print(f"sample[i]: {sample[i]}")
        print(f"mean[i]: {mean[i]}")
        print(f"diag_var[i]: {diag_var[i]}\n")

    print(f"final_pdf: {final_pdf}")

    # Compute the average PDF over the batch
    pdf = torch.prod(torch.tensor(final_pdf))

    return pdf

def diagonal_mvn_pdf(Z, mu, sigma_squared):
    """
    Calculate the PDF of a diagonal Multivariate Normal distribution at point Z.

    Args:
    - Z (torch.Tensor): The point at which to evaluate the PDF.
    - mu (torch.Tensor): The mean vector.
    - sigma_squared (torch.Tensor): The diagonal elements of the covariance matrix (variances).

    Returns:
    - pdf (torch.Tensor): The PDF value at point Z.
    """
    d = len(mu)  # Dimensionality of the data
    log_pdf = -0.5 * torch.sum(torch.log(2 * math.pi * sigma_squared) + ((Z - mu) ** 2) / sigma_squared)
    pdf = torch.exp(log_pdf)

    return pdf


if __name__ == '__main__':

        # # hidden state samples from N_\phi
        # samples = torch.tensor([
        #         [ 0.0317,  0.0163, -0.0062,  0.0694],
        #         [ 0.1244, -0.1245,  0.0240,  0.0954],
        #         [-0.0019,  0.2574,  0.0046,  0.1173],
        #         [ 0.0750,  0.1145,  0.0058,  0.0835],
        #         [-0.0746, -0.2696, -0.0636,  0.1118],
        #         [ 0.0324,  0.4568, -0.1179,  0.1117],
        #         [-0.0974,  0.3725,  0.2004,  0.1160],
        #         [ 0.0233,  0.1347, -0.1515,  0.1121],
        #         [-0.0045,  0.2049,  0.0881,  0.1196],
        #         [-0.0553,  0.2724, -0.1867,  0.0654],
        #         [ 0.1468,  0.4462, -0.1155,  0.1103],
        #         [-0.0465,  0.1672, -0.3725,  0.0757],
        #         [ 0.1774,  0.2232,  0.1803,  0.0605],
        #         [-0.0418,  0.0074, -0.0902,  0.0958],
        #         [-0.1014,  0.3710,  0.2454,  0.1107],
        #         [ 0.1449,  0.3199, -0.0995,  0.0704],
        #         [-0.0863,  0.5516, -0.0241,  0.0714],
        #         [ 0.0460,  0.0963,  0.2778,  0.1133],
        #         [-0.0719,  0.1996,  0.0647,  0.1074],
        #         [-0.1499,  0.4890,  0.1296,  0.1168],
        #         [-0.0507,  0.2351,  0.3182,  0.1121],
        #         [ 0.1518,  0.2855, -0.0937,  0.1177],
        #         [-0.1003,  0.0464, -0.3681,  0.1185],
        #         [-0.0153, -0.1498,  0.1894,  0.1344],
        #         [ 0.0205,  0.1959,  0.0646,  0.0846],
        #         [-0.1063, -0.0239, -0.0241,  0.1227],
        #         [ 0.1368,  0.1261, -0.1375,  0.0715],
        #         [ 0.2445, -0.0101, -0.2202,  0.1102],
        #         [-0.0744,  0.0426,  0.0086,  0.1157],
        #         [-0.0489,  0.1480,  0.1571,  0.0228],
        #         [ 0.0026, -0.0795, -0.0081,  0.0863],
        #         [-0.0049,  0.1559, -0.0120,  0.1167],
        #         [ 0.0007,  0.1708, -0.0445,  0.1118],
        #         [-0.1909,  0.2634, -0.3696,  0.1174],
        #         [ 0.0347,  0.3056, -0.1821,  0.1214],
        #         [ 0.0191,  0.1003, -0.0207,  0.1563],
        #         [ 0.0260,  0.6031,  0.1833,  0.1115],
        #         [-0.0414,  0.1826,  0.2671,  0.1292],
        #         [ 0.0965,  0.0700,  0.0541,  0.1377],
        #         [-0.0072,  0.2315, -0.1414,  0.1112],
        #         [-0.0538,  0.2213, -0.0701,  0.1122],
        #         [-0.2392,  0.4649,  0.4069,  0.1124],
        #         [ 0.1254,  0.0838, -0.0309,  0.1178],
        #         [-0.0612,  0.2077, -0.0963,  0.1192],
        #         [ 0.3067,  0.6233, -0.0608,  0.1154],
        #         [-0.1033,  0.1834,  0.1224,  0.0860],
        #         [ 0.0967,  0.1427, -0.0531,  0.1105],
        #         [-0.1522,  0.1413,  0.1760,  0.1499],
        #         [-0.0701,  0.1783,  0.0328,  0.0934],
        #         [-0.0528,  0.2753, -0.1241,  0.1411],
        #         [-0.0289, -0.0388,  0.0012,  0.1214],
        #         [-0.0326,  0.0397,  0.2994,  0.0119],
        #         [ 0.3858, -0.1019, -0.1145,  0.1201],
        #         [ 0.0562,  0.2904, -0.0739,  0.1191],
        #         [ 0.0283,  0.5290,  0.3120,  0.1099],
        #         [-0.2086,  0.1823, -0.0174,  0.1107],
        #         [-0.1716,  0.1652,  0.3463,  0.0643],
        #         [-0.0154,  0.3231,  0.3475,  0.1191],
        #         [-0.0102,  0.0883, -0.1109,  0.0574],
        #         [ 0.2360, -0.1470,  0.0131,  0.1718],
        #         [ 0.0185,  0.1974,  0.2239,  0.1114],
        #         [ 0.1559,  0.2052, -0.1978,  0.1238],
        #         [-0.1707,  0.4822,  0.0498,  0.1098],
        #         [ 0.1741,  0.0472,  0.1415,  0.1108]
        # ])

        # # predicted mean vectors for N_\phi
        # means = torch.tensor([
        #         [-0.0248,  0.1706,  0.0314,  0.1002],
        #         [-0.0248,  0.1706,  0.0314,  0.1002],
        #         [ 0.0253,  0.1559,  0.0055,  0.1171],
        #         [-0.0282,  0.1599,  0.0382,  0.1004],
        #         [ 0.0237,  0.1656,  0.0014,  0.1111],
        #         [ 0.0237,  0.1656,  0.0014,  0.1111],
        #         [ 0.0253,  0.1559,  0.0055,  0.1171],
        #         [ 0.0237,  0.1656,  0.0014,  0.1111],
        #         [-0.0282,  0.1599,  0.0382,  0.1004],
        #         [-0.0248,  0.1706,  0.0314,  0.1002],
        #         [ 0.0237,  0.1656,  0.0014,  0.1111],
        #         [-0.0282,  0.1599,  0.0382,  0.1004],
        #         [-0.0248,  0.1706,  0.0314,  0.1002],
        #         [-0.0282,  0.1599,  0.0382,  0.1004],
        #         [ 0.0237,  0.1656,  0.0014,  0.1111],
        #         [-0.0248,  0.1706,  0.0314,  0.1002],
        #         [-0.0282,  0.1599,  0.0382,  0.1004],
        #         [ 0.0253,  0.1559,  0.0055,  0.1171],
        #         [-0.0248,  0.1706,  0.0314,  0.1002],
        #         [ 0.0253,  0.1559,  0.0055,  0.1171],
        #         [ 0.0237,  0.1656,  0.0014,  0.1111],
        #         [ 0.0253,  0.1559,  0.0055,  0.1171],
        #         [ 0.0253,  0.1559,  0.0055,  0.1171],
        #         [-0.0282,  0.1599,  0.0382,  0.1004],
        #         [-0.0282,  0.1599,  0.0382,  0.1004],
        #         [-0.0248,  0.1706,  0.0314,  0.1002],
        #         [-0.0248,  0.1706,  0.0314,  0.1002],
        #         [ 0.0237,  0.1656,  0.0014,  0.1111],
        #         [ 0.0253,  0.1559,  0.0055,  0.1171],
        #         [-0.0282,  0.1599,  0.0382,  0.1004],
        #         [-0.0248,  0.1706,  0.0314,  0.1002],
        #         [ 0.0253,  0.1559,  0.0055,  0.1171],
        #         [ 0.0237,  0.1656,  0.0014,  0.1111],
        #         [ 0.0253,  0.1559,  0.0055,  0.1171],
        #         [ 0.0253,  0.1559,  0.0055,  0.1171],
        #         [-0.0248,  0.1706,  0.0314,  0.1002],
        #         [ 0.0237,  0.1656,  0.0014,  0.1111],
        #         [-0.0248,  0.1706,  0.0314,  0.1002],
        #         [-0.0282,  0.1599,  0.0382,  0.1004],
        #         [ 0.0237,  0.1656,  0.0014,  0.1111],
        #         [ 0.0237,  0.1656,  0.0014,  0.1111],
        #         [ 0.0237,  0.1656,  0.0014,  0.1111],
        #         [ 0.0253,  0.1559,  0.0055,  0.1171],
        #         [-0.0248,  0.1706,  0.0314,  0.1002],
        #         [ 0.0253,  0.1559,  0.0055,  0.1171],
        #         [-0.0282,  0.1599,  0.0382,  0.1004],
        #         [ 0.0237,  0.1656,  0.0014,  0.1111],
        #         [-0.0282,  0.1599,  0.0382,  0.1004],
        #         [-0.0282,  0.1599,  0.0382,  0.1004],
        #         [-0.0282,  0.1599,  0.0382,  0.1004],
        #         [-0.0282,  0.1599,  0.0382,  0.1004],
        #         [-0.0248,  0.1706,  0.0314,  0.1002],
        #         [ 0.0253,  0.1559,  0.0055,  0.1171],
        #         [ 0.0253,  0.1559,  0.0055,  0.1171],
        #         [ 0.0237,  0.1656,  0.0014,  0.1111],
        #         [ 0.0237,  0.1656,  0.0014,  0.1111],
        #         [-0.0248,  0.1706,  0.0314,  0.1002],
        #         [ 0.0253,  0.1559,  0.0055,  0.1171],
        #         [-0.0282,  0.1599,  0.0382,  0.1004],
        #         [-0.0282,  0.1599,  0.0382,  0.1004],
        #         [ 0.0237,  0.1656,  0.0014,  0.1111],
        #         [ 0.0253,  0.1559,  0.0055,  0.1171],
        #         [ 0.0237,  0.1656,  0.0014,  0.1111],
        #         [ 0.0237,  0.1656,  0.0014,  0.1111]
        # ])

        # # predicted variances for N_\phi
        # diag_vars = torch.tensor([
        #         [8.1372e-03, 2.6815e-02, 3.7143e-02, 1.4694e-03],
        #         [8.1372e-03, 2.6815e-02, 3.7143e-02, 1.4694e-03],
        #         [2.0956e-02, 6.7220e-02, 2.8870e-02, 9.5371e-06],
        #         [7.9376e-03, 2.8334e-02, 3.5356e-02, 1.3230e-03],
        #         [1.7298e-02, 6.7676e-02, 3.3873e-02, 8.5967e-07],
        #         [1.7298e-02, 6.7676e-02, 3.3873e-02, 8.5967e-07],
        #         [2.0956e-02, 6.7220e-02, 2.8870e-02, 9.5371e-06],
        #         [1.7298e-02, 6.7676e-02, 3.3873e-02, 8.5967e-07],
        #         [7.9376e-03, 2.8334e-02, 3.5356e-02, 1.3230e-03],
        #         [8.1372e-03, 2.6815e-02, 3.7143e-02, 1.4694e-03],
        #         [1.7298e-02, 6.7676e-02, 3.3873e-02, 8.5967e-07],
        #         [7.9376e-03, 2.8334e-02, 3.5356e-02, 1.3230e-03],
        #         [8.1372e-03, 2.6815e-02, 3.7143e-02, 1.4694e-03],
        #         [7.9376e-03, 2.8334e-02, 3.5356e-02, 1.3230e-03],
        #         [1.7298e-02, 6.7676e-02, 3.3873e-02, 8.5967e-07],
        #         [8.1372e-03, 2.6815e-02, 3.7143e-02, 1.4694e-03],
        #         [7.9376e-03, 2.8334e-02, 3.5356e-02, 1.3230e-03],
        #         [2.0956e-02, 6.7220e-02, 2.8870e-02, 9.5371e-06],
        #         [8.1372e-03, 2.6815e-02, 3.7143e-02, 1.4694e-03],
        #         [2.0956e-02, 6.7220e-02, 2.8870e-02, 9.5371e-06],
        #         [1.7298e-02, 6.7676e-02, 3.3873e-02, 8.5967e-07],
        #         [2.0956e-02, 6.7220e-02, 2.8870e-02, 9.5371e-06],
        #         [2.0956e-02, 6.7220e-02, 2.8870e-02, 9.5371e-06],
        #         [7.9376e-03, 2.8334e-02, 3.5356e-02, 1.3230e-03],
        #         [7.9376e-03, 2.8334e-02, 3.5356e-02, 1.3230e-03],
        #         [8.1372e-03, 2.6815e-02, 3.7143e-02, 1.4694e-03],
        #         [8.1372e-03, 2.6815e-02, 3.7143e-02, 1.4694e-03],
        #         [1.7298e-02, 6.7676e-02, 3.3873e-02, 8.5967e-07],
        #         [2.0956e-02, 6.7220e-02, 2.8870e-02, 9.5371e-06],
        #         [7.9376e-03, 2.8334e-02, 3.5356e-02, 1.3230e-03],
        #         [8.1372e-03, 2.6815e-02, 3.7143e-02, 1.4694e-03],
        #         [2.0956e-02, 6.7220e-02, 2.8870e-02, 9.5371e-06],
        #         [1.7298e-02, 6.7676e-02, 3.3873e-02, 8.5967e-07],
        #         [2.0956e-02, 6.7220e-02, 2.8870e-02, 9.5371e-06],
        #         [2.0956e-02, 6.7220e-02, 2.8870e-02, 9.5371e-06],
        #         [8.1372e-03, 2.6815e-02, 3.7143e-02, 1.4694e-03],
        #         [1.7298e-02, 6.7676e-02, 3.3873e-02, 8.5967e-07],
        #         [8.1372e-03, 2.6815e-02, 3.7143e-02, 1.4694e-03],
        #         [7.9376e-03, 2.8334e-02, 3.5356e-02, 1.3230e-03],
        #         [1.7298e-02, 6.7676e-02, 3.3873e-02, 8.5967e-07],
        #         [1.7298e-02, 6.7676e-02, 3.3873e-02, 8.5967e-07],
        #         [1.7298e-02, 6.7676e-02, 3.3873e-02, 8.5967e-07],
        #         [2.0956e-02, 6.7220e-02, 2.8870e-02, 9.5371e-06],
        #         [8.1372e-03, 2.6815e-02, 3.7143e-02, 1.4694e-03],
        #         [2.0956e-02, 6.7220e-02, 2.8870e-02, 9.5371e-06],
        #         [7.9376e-03, 2.8334e-02, 3.5356e-02, 1.3230e-03],
        #         [1.7298e-02, 6.7676e-02, 3.3873e-02, 8.5967e-07],
        #         [7.9376e-03, 2.8334e-02, 3.5356e-02, 1.3230e-03],
        #         [7.9376e-03, 2.8334e-02, 3.5356e-02, 1.3230e-03],
        #         [7.9376e-03, 2.8334e-02, 3.5356e-02, 1.3230e-03],
        #         [7.9376e-03, 2.8334e-02, 3.5356e-02, 1.3230e-03],
        #         [8.1372e-03, 2.6815e-02, 3.7143e-02, 1.4694e-03],
        #         [2.0956e-02, 6.7220e-02, 2.8870e-02, 9.5371e-06],
        #         [2.0956e-02, 6.7220e-02, 2.8870e-02, 9.5371e-06],
        #         [1.7298e-02, 6.7676e-02, 3.3873e-02, 8.5967e-07],
        #         [1.7298e-02, 6.7676e-02, 3.3873e-02, 8.5967e-07],
        #         [8.1372e-03, 2.6815e-02, 3.7143e-02, 1.4694e-03],
        #         [2.0956e-02, 6.7220e-02, 2.8870e-02, 9.5371e-06],
        #         [7.9376e-03, 2.8334e-02, 3.5356e-02, 1.3230e-03],
        #         [7.9376e-03, 2.8334e-02, 3.5356e-02, 1.3230e-03],
        #         [1.7298e-02, 6.7676e-02, 3.3873e-02, 8.5967e-07],
        #         [2.0956e-02, 6.7220e-02, 2.8870e-02, 9.5371e-06],
        #         [1.7298e-02, 6.7676e-02, 3.3873e-02, 8.5967e-07],
        #         [1.7298e-02, 6.7676e-02, 3.3873e-02, 8.5967e-07]
        # ])

        # # pdf = diag_mvg_pdf_torch(samples, means, diag_vars)
        # pdf = diagonal_mvn_pdf(samples[0], means[0], diag_vars[0])

        # print(f"\npdf: {pdf}\n")

        # # Define mean vector and diagonal elements of the covariance matrix
        # mu = torch.tensor([2.0, 3.0, 1.0, 2.5])
        # sigma_squared = torch.tensor([0.5, 1.0, 0.8, 0.3])

        # # Sample point Z
        # Z = torch.tensor([2.2, 2.8, 0.9, 0.5])


        # Define mean vector and diagonal elements of the covariance matrix
        mu = torch.tensor([-0.0248,  0.1706,  0.0314,  0.1002])
        sigma_squared = torch.tensor([8.1372e-03, 2.6815e-02, 3.7143e-02, 1.4694e-03])

        # Sample point Z
        Z = torch.tensor([0.0317,  0.0163, -0.0062,  0.0694])

        # Calculate the PDF
        # pdf = diagonal_mvn_pdf(Z, mu, sigma_squared)
        pdf = diag_mvg_pdf_from_scratch(Z, mu, sigma_squared)
        # print("PDF at Z:", pdf.item())
        print("PDF at Z:", pdf)