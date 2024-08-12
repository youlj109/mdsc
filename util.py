import torch
from sklearn.cluster import KMeans

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
initial_centers = None

def ComputeKmeans(V_i, k):
    global initial_centers
    if initial_centers is None:
        kmeans = KMeans(n_clusters=k, n_init=1)
    else:
        kmeans = KMeans(n_clusters=k, init=initial_centers, n_init=1)

    kmeans.fit(V_i.cpu().detach().numpy())  
    initial_centers = kmeans.cluster_centers_
    centroids = torch.tensor(kmeans.cluster_centers_, device=DEVICE)  
    labels = torch.tensor(kmeans.labels_, device=DEVICE)  
    distances = torch.norm(V_i.unsqueeze(1) - centroids[labels.long()], dim=2)  
    mean_distances = torch.zeros(k, device=DEVICE)
    fai = torch.zeros(k, device=DEVICE)
    for i in range(k):
        distances_i = distances[labels == i]
        mean_distances[i] = torch.mean(distances_i)
        fai[i] = mean_distances[i] / torch.log(torch.tensor(len(distances_i)) + 10) 
    return centroids[labels.long()], fai[labels.long()]


def ComputeCosinSim(X, Y):
    norm_X = torch.norm(X, dim=1, keepdim=True)
    norm_Y = torch.norm(Y, dim=1, keepdim=True)

    cosine_similarity = torch.mm(X, Y.t()) / (norm_X * norm_Y.t()+0.0001)
    return cosine_similarity


def ProtoNCE_1(X_t, W_t, W_t_1,tao):
    V_i = torch.matmul(X_t,W_t)
    V_i_ = torch.matmul(X_t,W_t_1)
    numerator_1 = torch.exp(torch.diag(ComputeCosinSim(V_i, V_i_)) / tao)
    fenmu_1 = torch.sum(torch.exp(ComputeCosinSim(V_i, V_i_) / tao), dim=1)

    return (-torch.log(numerator_1 / fenmu_1)).sum()

def ProtoNCE_2(X_t, W_t,M):
    V_i = torch.matmul(X_t,W_t)

    k = M
    
    if initial_centers is not None:
            V_i = torch.vstack([V_i, torch.tensor(initial_centers,device = DEVICE)])
            
    centroids, fai = ComputeKmeans(V_i, k)
         
    numerator_2_k = torch.exp(torch.diag(ComputeCosinSim(V_i, centroids) / (0.1+fai.unsqueeze(0)))) 
    fenmu_2_k = torch.sum(torch.exp(ComputeCosinSim(V_i, centroids) / (0.1+fai.unsqueeze(0))), dim=1)
    loss_M = -torch.log((numerator_2_k / fenmu_2_k))

    return loss_M.sum()

def gfun(X_t, W_t, k):
    t  = 1
    norms = torch.norm(X_t[:, None] - X_t, dim=2)
    top_k_values, _ = torch.topk(norms, k, dim=1, largest=False)
    G = torch.where(norms <= top_k_values[:, -1][:, None], torch.exp(-norms/t), torch.zeros_like(norms))
    D = torch.diag(torch.sum(G, dim=1))
    L = D - G
    result = torch.trace(torch.matmul(torch.matmul(torch.matmul(X_t, W_t).t(), L), torch.matmul(X_t, W_t)))
    return result
    
def rfun(W_t):
    return torch.sum(torch.norm(W_t, p=2, dim=1))

def F_norm_2(W):
    return torch.sum(torch.abs(W)**2)

def Lt(X_t, W_t, W_t_1, nanpta1, nanpta2, nanpta3,nanpta4,tao,M,k):
    I_m = torch.eye(W_t.shape[1], device=W_t.device)
    loss = ProtoNCE_1(X_t, W_t, W_t_1,tao) +nanpta1*ProtoNCE_2(X_t, W_t,M) + nanpta2*gfun(X_t,W_t,k) + nanpta3*rfun(W_t) + nanpta4*F_norm_2(torch.matmul(W_t.T, W_t) - I_m)
    return loss