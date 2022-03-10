import torch
import torch.nn as nn
import torch.nn.functional as F



'''
Oriinal code from github
'''

class ScoreEstimator:
    def __init__(self):
        pass

    def rbf_kernel(self, x1, x2, kernel_width):
        return torch.exp(
            -torch.sum(torch.mul((x1 - x2), (x1 - x2)), dim=-1) / (2 * torch.mul(kernel_width, kernel_width))
        )

    def gram(self, x1, x2, kernel_width):
        x_row = torch.unsqueeze(x1, -2)
        x_col = torch.unsqueeze(x2, -3)
        kernel_width = kernel_width[..., None, None]
        return self.rbf_kernel(x_row, x_col, kernel_width)

    def grad_gram(self, x1, x2, kernel_width):
        x_row = torch.unsqueeze(x1, -2)
        x_col = torch.unsqueeze(x2, -3)
        kernel_width = kernel_width[..., None, None]
        G = self.rbf_kernel(x_row, x_col, kernel_width)
        diff = (x_row - x_col) / (kernel_width[..., None] ** 2)
        G_expand = torch.unsqueeze(G, -1)
        grad_x2 = G_expand * diff
        grad_x1 = G_expand * (-diff)
        return G, grad_x1, grad_x2

    def heuristic_kernel_width(self, x_samples, x_basis):
        n_samples = x_samples.size()[-2]
        n_basis = x_basis.size()[-2]
        x_samples_expand = torch.unsqueeze(x_samples, -2)
        x_basis_expand = torch.unsqueeze(x_basis, -3)
        pairwise_dist = torch.sqrt(
            torch.sum(torch.mul(x_samples_expand - x_basis_expand, x_samples_expand - x_basis_expand), dim=-1)
        )
        k = n_samples * n_basis // 2
        top_k_values = torch.topk(torch.reshape(pairwise_dist, [-1, n_samples * n_basis]), k=k)[0]
        kernel_width = torch.reshape(top_k_values[:, -1], x_samples.size()[:-2])
        return kernel_width.detach()

    def compute_gradients(self, samples, x=None):
        raise NotImplementedError()


class SpectralScoreEstimator(ScoreEstimator):
    def __init__(self, n_eigen=None, eta=None, n_eigen_threshold=None):
        self._n_eigen = n_eigen
        self._eta = eta
        self._n_eigen_threshold = n_eigen_threshold
        super().__init__()

    def nystrom_ext(self, samples, x, eigen_vectors, eigen_values, kernel_width):
        M = torch.tensor(samples.size()[-2]).to(samples.device)
        Kxq = self.gram(x, samples, kernel_width)
        ret = torch.sqrt(M.float()) * torch.matmul(Kxq, eigen_vectors)
        ret *= 1. / torch.unsqueeze(eigen_values, dim=-2)
        return ret

    def compute_gradients(self, samples, x=None):
        if x is None:
            kernel_width = self.heuristic_kernel_width(samples, samples)
            x = samples
        else:
            _samples = torch.cat([samples, x], dim=-2)
            kernel_width = self.heuristic_kernel_width(_samples, _samples)

        M = samples.size()[-2]
        Kq, grad_K1, grad_K2 = self.grad_gram(samples, samples, kernel_width)
        if self._eta is not None:
            Kq += self._eta * torch.eye(M)

        # eigen_values, eigen_vectors = torch.symeig(Kq, eigenvectors=True, upper=True)
        eigen_values, eigen_vectors =  torch.linalg.eigh(Kq, UPLO='U')
        if (self._n_eigen is None) and (self._n_eigen_threshold is not None):
            eigen_arr = torch.mean(
                torch.reshape(eigen_values, [-1, M]), dim=0)

            eigen_arr = torch.flip(eigen_arr, [-1])
            eigen_arr /= torch.sum(eigen_arr)
            eigen_cum = torch.cumsum(eigen_arr, dim=-1)
            eigen_lt = torch.lt(eigen_cum, self._n_eigen_threshold)
            self._n_eigen = torch.sum(eigen_lt)
        if self._n_eigen is not None:
            eigen_values = eigen_values[..., -self._n_eigen:]
            eigen_vectors = eigen_vectors[..., -self._n_eigen:]
        eigen_ext = self.nystrom_ext(samples, x, eigen_vectors, eigen_values, kernel_width)
        grad_K1_avg = torch.mean(grad_K1, dim=-3)
        M = torch.tensor(M).to(samples.device)
        beta = -torch.sqrt(M.float()) * torch.matmul(torch.transpose(eigen_vectors, -1, -2),
                                                     grad_K1_avg) / torch.unsqueeze(eigen_values, -1)
        grads = torch.matmul(eigen_ext, beta)
        self._n_eigen = None
        return grads

def entropy_surrogate(estimator, samples):
    dlog_q = estimator.compute_gradients(samples.detach(), None)
    surrogate_cost = torch.mean(torch.sum(dlog_q.detach() * samples, -1))
    return surrogate_cost

'''
Oriinal code from github, noise is the only modification
'''
def GenerateSamples(x, z, device, noise_mag):
    num_sample, d = z.size()
    noise = noise_mag * torch.randn(num_sample, d).to(device)
    z_noise = z + noise
    x_z = torch.cat([x, z_noise], dim=-1)
    # x_z = x_z.to(device)
    # z_noise = z_noise.to(device)
    return x_z, x, z_noise


'''
My code
'''
# def MIGE_prev(x, z, GenerateData, learning_rate, device,noise_mag=0.1, threshold=None, n_eigen=None):
#     spectral_j = SpectralScoreEstimator(n_eigen=n_eigen, n_eigen_threshold=threshold)
#     spectral_m = SpectralScoreEstimator(n_eigen=n_eigen, n_eigen_threshold=threshold)
#     x_z, _, z_noise = GenerateData(x, z, device,noise_mag)
#     # negative mutual information
#     ungrad_mutual = -entropy_surrogate(spectral_j, x_z) \
#       + entropy_surrogate(spectral_m, z_noise)
#     # grad_mutual.backward(retain_graph = True)
#     return ungrad_mutual
# def MIGE_compute(grad_mutual):
#     grad_mutual.backward(retain_graph = True)
#     return


def MIGE(x, z, GenerateData, learning_rate, beta,device,  noise_mag=0.1, threshold=None,n_eigen=None):
    spectral_j = SpectralScoreEstimator(n_eigen=n_eigen, n_eigen_threshold=threshold)
    spectral_m = SpectralScoreEstimator(n_eigen=n_eigen, n_eigen_threshold=threshold)
    x_z, _, z_noise = GenerateData(x, z, device,noise_mag)
    # negative mutual information
    grad_mutual = (- entropy_surrogate(spectral_j, x_z) \
      + entropy_surrogate(spectral_m, z_noise)) * beta
    grad_mutual.backward(retain_graph = True)
    return
    # with torch.no_grad():
    #     for p in ae_model.parameters():
    #         # print(p.grad)
    #         # print(learning_rate)
    #         new_val = update_function(p, p.grad, learning_rate)
    #         p.copy_(new_val)


# def MIGE_update(ae_model, x, z, GenerateData, learning_rate, device, threshold=None, n_eigen=None):
#     spectral_j = SpectralScoreEstimator(n_eigen=n_eigen, n_eigen_threshold=threshold)
#     spectral_m = SpectralScoreEstimator(n_eigen=n_eigen, n_eigen_threshold=threshold)
#     x_z, _, z_noise = GenerateData(x, z, device)
#     # negative mutual information
#     grad_mutual = -entropy_surrogate(spectral_j, x_z) \
#       + entropy_surrogate(spectral_m, z_noise)
#     grad_mutual.backward()
#
#     with torch.no_grad():
#         for p in ae_model.parameters():
#             # print(p.grad)
#             # print(learning_rate)
#             new_val = update_function(p, p.grad, learning_rate)
#             p.copy_(new_val)
#     return ae_model

    # for rho in range_rho:
    #     rho = torch.FloatTensor([rho])
    #     rho.requires_grad = True
    #     xs_ys, xs, ys = GenerateData(d, rho, num_sample)

    #     ans = entropy_surrogate(spectral_j, xs_ys) \
    #           - entropy_surrogate(spectral_m, ys)

    #     ans.backward()
    #     approximations.append(rho.grad.data)

    # approximations = torch.stack(approximations).view(-1).detach().cpu().numpy()
