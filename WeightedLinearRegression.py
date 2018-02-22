import numpy as np
import matplotlib.pyplot as plt


class WeightedLinReg:
    
    def __init__(self, input_file, output_file):
        self.x = self.ReadLinFile(input_file)
        self.y = self.ReadLinFile(output_file)
        self.num_examples = self.x.shape[0]
        self.theta = np.empty(2)
        self.tau = 0.8
    
    #function to read file with only 1 feature
    def ReadLinFile(self, file_name):
        fin = open(file_name, 'r')
        data = []
        for inp in fin:
            data.append(float(inp[:-1]))
        return np.array(data).reshape((len(data),1))
    
    #function to normalize data
    def NormalizeData(self):
        mu = np.mean(self.x)
        sigma = np.std(self.x)
        self.x = (self.x-mu)/sigma

    #function to solve normal equation (theta = ((X_T.X)^-1).X_T.Y ) for unweighted linear regression
    def SolveNormalEquation(self, weights=None):
        if(weights==None):
            x = np.c_[np.ones((self.num_examples,1)),self.x]
            self.theta = np.linalg.inv(np.dot(x.T,x)).dot(x.T).dot(self.y)
        
        return self.theta
    
    #function to solve generalized normal equation (theta = ((X_T.W.X)^-1).X_T.W.Y ) for weighted linear regression
    def SolveGeneralNormalEquation(self):
        test_xs = np.linspace(-2,2,num=100)
        test_ys = np.empty(test_xs.shape)
        trainX = np.c_[np.ones((self.num_examples,1)),self.x]
        for i,testX in enumerate(test_xs):
            W=np.diag(np.exp(-((trainX[:,1] - testX)**2)/(2*self.tau**2)))
            theta = np.dot(np.linalg.inv(trainX.T.dot(W).dot(trainX)),trainX.T.dot(W).dot(self.y))
            test_ys[i] = theta[1]*testX + theta[0]
        return (test_xs,test_ys)
    
def plot(data, plot_type, title="", options=[]):
    if plot_type == "scatter":                 #scatter plot. 'data' has (x,y) pairs. 'options' is empty
        plt.scatter(data[0], data[1], s=5)
        plt.scatter(data[0], data[1], s=5)
    elif plot_type == "equation":              #equation plot. 'data' has equation string. 'options' has domain of 'x'
        x=np.linspace(options[0],options[1],100)
        plt.plot(x, eval(data))
    elif plot_type == "scatterequation":       #equation and scatter plot in same figure. 'data' has (x,y) pairs and equation string. 'options' has domain of 'x'
        plt.scatter(data[0], data[1], s=5)
        x=np.linspace(options[0],options[1],100)
        plt.plot(x, eval(data[2]))
    elif plot_type == "scatterequationdata":   #curve using (x,y) pairs and scatter plot in same figure. 'data' has (x,y) pairs of scatter plot and (x,y) pairs of curve
        plt.scatter(data[0], data[1], s=5)
        plt.plot(data[2], data[3])
    plt.title(title)
    plt.show()

if __name__=='__main__':
    
    #create a weighted linear regression object
    if(len(sys.argv)==3):
        lr = WeightedLinReg(sys.argv[1],sys.argv[2])
    else:
        lr = WeightedLinReg("weightedX.csv","weightedY.csv")
    
    #normalize data to 0 mean and 1 standard deviation
    lr.NormalizeData()
    
    #compute parameters by solving normal equation for unweighted regression
    c,m = lr.SolveNormalEquation()         #y = mx + c
    c,m=c[0],m[0]
    print("y = "+str(m)+"x + "+str(c))
    
    #plot scatter points along with regression line
    plot([lr.x,lr.y,str(m)+"*x + "+str(c)],"scatterequation","UnweightedRegression",options=[-2,2])
    
    #compute parameters by solving generalized normal equation for weighted regression
    data = lr.SolveGeneralNormalEquation()
    
    plot([lr.x,lr.y,data[0],data[1]],"scatterequationdata", "WeightedRegression (tau="+str(lr.tau)+")")
    
    #plot weighted regression output for different bandwidths
    taus = [0.1,0.3,2,10]
    for tau in taus:
        lr.tau = tau
        data = lr.SolveGeneralNormalEquation()
        plot([lr.x,lr.y,data[0],data[1]],"scatterequationdata", "WeightedRegression (tau="+str(lr.tau)+")")