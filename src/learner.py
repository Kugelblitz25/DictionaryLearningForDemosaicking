import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor 

class Dictionary:
    def __init__(self, l: int, k: int, c: int):
        self.l = l
        self.k = k
        self.c = c
        self.D = 255*np.random.rand(l, k)
        self.A = np.random.rand(k, c)
        self.A = self.A/np.sum(self.A, axis=0)

    def learn(self, Z: np.ndarray, maxIters: int = 50, lambda_: float = 0.01):
        for i in range(maxIters):
            print(f"Iteration: {i+1}")
            self.updateA(Z, lambda_)
            prevD = self.D.copy()
            self.updateD(Z, lambda_)
            print(f"Error: {np.linalg.norm(prevD - self.D)}")
            if np.linalg.norm(prevD - self.D)< 0.2:
                break

    def updateAj(self, j):
        G = self.Z[:, j].reshape(-1, 1)@np.ones((1, self.k)) - self.D 
        P = np.diag(((self.D - self.Z[:,j].reshape(-1,1))**2).sum(axis=0))
        bj = np.linalg.inv(G.T@G + self.lambda_*P)@np.ones((self.k,1))
        return bj.reshape(-1,)/np.sum(bj)

    def updateA(self, Z: np.ndarray, lambda_: float):
        self.Z = Z
        self.lambda_ = lambda_
        for j in tqdm(range(self.c), desc='Updating A'):
            result = self.updateAj(j)
            self.A[:, j] = result

    def updateD(self, Z: np.ndarray, lambda_: float):
        print("Updating D")
        R = np.zeros((self.k, self.k))
        S = np.zeros((self.k, self.l))
        for j in range(self.c):
            R += self.A[:, j].reshape(-1,1)@self.A[:, j].reshape(1,-1)+lambda_*np.diag(self.A[:, j]**2)
            S += (self.A[:, j]+self.A[:, j]**2).reshape(-1,1)@Z[:, j].reshape(1,-1)

        self.D = (np.linalg.inv(R)@S).T