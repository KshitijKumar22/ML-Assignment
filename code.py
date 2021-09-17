import numpy as np

hyperParameter = 100000 #Taking lambda as 1 lakh

def cost(X, Y, theta):
    hypo = np.dot(X, theta)
    err = (hypo - Y) ** 2
    cost = np.mean(err)
    return cost

#WITHOUT REGULARIZATION

def batch_GDA(train_X, train_Y, theta, num_Iters, learn_Rate):
    m = train_X.shape[0]
    history_Lt = []
    for i in range(num_Iters):
        gradFactor = (1 /m) * (np.dot(train_X, theta) - train_Y).dot(train_X)
        theta = theta - learn_Rate * gradFactor

        history_Lt.append(cost(train_X, train_Y, theta))

    return history_Lt , theta


def stochastic_GDA(train_X, train_Y, theta, num_Iters, learn_Rate):
    m = train_X.shape[0]
    errList = []
    for i in range(num_Iters):
        for j in range(m):
            idx = np.random.randint(m)
            X = train_X[idx:idx+1, :]
            Y = train_Y[idx:idx+1]

            gradFactor = (1/m)*(np.dot(X, theta) - Y).dot(X)
            theta = theta - learn_Rate * gradFactor

        errList.append(cost(train_X,train_Y, theta))

    return errList, theta


def mini_batch_GDA(train_X,train_Y,theta,num_Iters,learn_Rate):
    m=train_X.shape[0]
    SizeofBatch = 45
    errList = []
    for i in range(0,num_Iters):
        for j in range(0,(int)(m/45)):
            left = 45*j
            right = 45*j + 45
            if right>m:
                right = m
            X = train_X[left:right]
            Y = train_Y[left:right]
            hypo = np.dot(X,theta)
            err = hypo - Y
            XTranspose = np.transpose(X)
            gradFactor = np.dot(XTranspose,err)/m
            theta = theta - learn_Rate*gradFactor
        errList.append(cost(train_X,train_Y,theta))
    return errList,theta

#WITH REGULARIZATION

def batch_GDA_reg(train_X, train_Y, theta, num_Iters, learn_Rate):
    m = train_X.shape[0]
    history_Lt = []
    for i in range(num_Iters):
        gradFactor = (1 /m) * (np.dot(train_X, theta) - train_Y).dot(train_X)
        theta = (1-(learn_Rate*hyperParameter)/m)*theta - learn_Rate * gradFactor #SEE THE CHANGE IN THIS LINE

        history_Lt.append(cost(train_X, train_Y, theta))

    return history_Lt , theta


def stochastic_GDA_reg(train_X, train_Y, theta, num_Iters, learn_Rate):
    m = train_X.shape[0]
    errList = []
    for i in range(num_Iters):
        for j in range(m):
            idx = np.random.randint(m)
            X = train_X[idx:idx+1, :]
            Y = train_Y[idx:idx+1]

            gradFactor = (1/m)*(np.dot(X, theta) - Y).dot(X)
            theta =(1-(learn_Rate*hyperParameter)/m)*theta - learn_Rate * gradFactor #SEE THE CHANGE IN THIS LINE

        errList.append(cost(train_X,train_Y, theta))

    return errList, theta


def mini_batch_GDA_reg(train_X,train_Y,theta,num_Iters,learn_Rate):
    m=train_X.shape[0]
    SizeofBatch = 45
    errList = []
    for i in range(0,num_Iters):
        for j in range(0,(int)(m/45)):
            left = 45*j
            right = 45*j + 45
            if right>m:
                right = m
            X = train_X[left:right]
            Y = train_Y[left:right]
            hypo = np.dot(X,theta)
            err = hypo - Y
            XTranspose = np.transpose(X)
            gradFactor = np.dot(XTranspose,err)/m
            theta = (1-(learn_Rate*hyperParameter)/m)*theta - learn_Rate*gradFactor #SEE THE CHANGE IN THIS LINE
        errList.append(cost(train_X,train_Y,theta))
    return errList,theta
