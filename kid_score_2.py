import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input


inception = InceptionV3(include_top=True, weights='imagenet')

def kid_score(X, Y):
    """
    Given X, Y (numpy) batches of inception outputs of generated and real images,
    return the KID score.
    """

    n = X.shape[0]
    m = Y.shape[0]

    def k(x, y):
        return (1/x.shape[0]*np.dot(x, y)+1)**(1/3)

    def f(X, Y):
        # First 2 sums. We use the fact that k is symmetric
        res = 0
        for i in range(n):
            for j in range(i+1, m):
                res += k(X(i), Y(j))
        
        return 2*res
    
    def f2(X, Y):
        # Third sum.
        res = 0
        for i in range(n):
            for j in range(m):
                res += k(X(i), Y(j))
        
        return res

    return 1/(n*(n-1))*f(X, X) + 1/(m*(m-1))*f(Y, Y) + 2/(n*m)*f2(X, Y)

def KID(images_gen, images_real):
    # TODO check outputs format. Images should be between -1 and 1
    # (we can use preprocess input to go from [0, 255] to [-1, 1])
    preds_gen = inception.predict(images_gen)
    preds_real = inception.predict(images_gen)
    return kid_score(preds_gen, preds_real)
