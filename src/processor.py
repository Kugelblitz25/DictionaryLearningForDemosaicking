import numpy as np
import cv2 as cv
from tqdm import tqdm


class Processor:
    def __init__(self, depth: int = 1):
        self.depth = depth

    def interpolate(self, img: np.ndarray, 
                          alpha: float = 0.5, 
                          beta: float = 5/8, 
                          gamma: float = 0.75) -> np.ndarray:
        interpKernelRB = np.array([[0.25, 0.5, 0.25],
                                   [0.5, 1, 0.5],
                                   [0.25, 0.5, 0.25]])
        interpKernelG = np.array([[0, 0.25, 0],
                                  [0.25, 1, 0.25],
                                  [0, 0.25, 0]])

        gInterp = cv.filter2D(img[:, :, 1], -1, interpKernelG)
        bInterp = cv.filter2D(img[:, :, 2], -1, interpKernelRB)
        rInterp = cv.filter2D(img[:, :, 0], -1, interpKernelRB)

        gradKernelRB = np.array([[0, 0, -0.25, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [-0.25, 0, 1, 0, -0.25],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, -0.25, 0, 0]])
        gradKernelG = np.array([[0, 0, -1/8, 0, 0],
                                [0, -1/8, 0, -1/8, 0],
                                [-1/8, 0, 1, 0, -1/8],
                                [0, -1/8, 0, -1/8, 0],
                                [0, 0, -1/8, 0, 0]])

        gradR = cv.filter2D(img[:, :, 0], -1, gradKernelRB)
        gradB = cv.filter2D(img[:, :, 2], -1, gradKernelRB)
        gradG = cv.filter2D(img[:, :, 1], -1, gradKernelG)

        R = rInterp + beta * gradG + gamma * gradB
        G = gInterp + alpha * gradR + alpha * gradB
        B = bInterp + beta * gradG + gamma * gradR

        newImg = np.array([R, G, B]).astype('uint8').transpose(1, 2, 0)
        return newImg

    def downsample(self, img: np.ndarray, r: int = 2) -> np.ndarray:
        return cv.resize(img, (img.shape[1]//r, img.shape[0]//r), interpolation=cv.INTER_LANCZOS4)

    def getZ(self, Y: np.ndarray, X: np.ndarray, b: int = 15) -> np.ndarray:
        assert Y.shape == X.shape, 'Y and X must have the same shape'
        
        h, w, c = Y.shape
        n = (h-b)*(w-b)
        Z = np.zeros((2*b*b*c, n))
        for i in range(h-b):
            for j in range(w-b):
                Z[:, i*(w-b) + j] = np.concatenate((Y[i:i+b, j:j+b, :].flatten(), X[i:i+b, j:j+b, :].flatten()))

        return Z
    
    def getCoefficients(self, D: np.ndarray, X: np.ndarray, lambda_: float = 0.1) -> np.ndarray:
        l, k = D.shape
        c = X.shape[1]
        A = np.zeros((k, c))
        for j in tqdm(range(c), desc="Calculating Aj"):
            G = X[:,j].reshape(-1, 1)@np.ones((1, k)) - D
            P = np.diag(((D - X[:,j].reshape(-1,1))**2).sum(axis=0))
            bj = np.linalg.inv(G.T@G + lambda_*P)@np.ones((k,1))
            A[:, j] = bj.reshape(-1,)/np.sum(bj)
        return A
    
    def dictionaryCorrection(self, D: np.ndarray, 
                                   img: np.ndarray, 
                                   b: int = 15, 
                                   lambda_: float = 0.1,
                                   sigma: float = 0.7) -> np.ndarray:
        h, w, c = img.shape
        n = (h-b)*(w-b)
        X = np.zeros((b*b*3, n))
        for i in range(h-b):
            for j in range(w-b):
                X[:, i*(w-b) + j] = img[i:i+b, j:j+b, :].flatten()

        DX = D[b*b*3:, :] 
        print("Calculating A for reconstruction")
        A = self.getCoefficients(DX, X, lambda_)
        DY = D[:b*b*3, :]
        Y = DY@A
        newImg = np.zeros_like(img, dtype='float64')
        g = cv.getGaussianKernel(b, sigma)
        kernel = np.expand_dims(g@g.T, axis=2)
        kernel = kernel/np.sum(kernel)
        print("Reconstructing Image")
        for i in range(h-b):
            for j in range(w-b):
                patch = Y[:, i*(w-b) + j].reshape(b, b, 3)
                newImg[i:i+b, j:j+b, :] += patch*kernel
        return newImg.astype('uint8')