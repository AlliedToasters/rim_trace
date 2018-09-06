import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors
from math import sqrt, pi

def calc_R(x,y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()

def sigma(coords, x, y, r):
    """Computes Sigma for circle fit."""
    dx, dy, sum_ = 0., 0., 0.

    for i in range(len(coords)):
        dx = coords[i][1] - x;
        dy = coords[i][0] - y;
        sum_ += (sqrt(dx*dx+dy*dy) - r)**2;
    return sqrt(sum_/len(coords));

def hyper_fit(coords, IterMax=99, verbose=False):
    """
    Fits coords to circle.
    
    Coords is a list or numpy array with len>2 of the form:
        [
    [x_coord, y_coord],
    ...,
    [x_coord, y_coord]
    ]
  
    for observed data points in a circle to be fit.
    Returns x, y, and r for best-fit circle.
    
    """
    n = len(coords)
    X = np.array([x[0] for x in coords])
    Y = np.array([x[1] for x in coords])
    Xi = X - X.mean()
    Yi = Y - Y.mean()
    Zi = Xi*Xi + Yi*Yi
    
    #compute moments
    Mxy = (Xi*Yi).sum()/n;
    Mxx = (Xi*Xi).sum()/n;
    Myy = (Yi*Yi).sum()/n;
    Mxz = (Xi*Zi).sum()/n;
    Myz = (Yi*Zi).sum()/n;
    Mzz = (Zi*Zi).sum()/n;
    
    #computing the coefficients of characteristic polynomial
    Mz = Mxx + Myy;
    Cov_xy = Mxx*Myy - Mxy*Mxy;
    Var_z = Mzz - Mz*Mz;

    A2 = 4*Cov_xy - 3*Mz*Mz - Mzz;
    A1 = Var_z*Mz + 4.*Cov_xy*Mz - Mxz*Mxz - Myz*Myz;
    A0 = Mxz*(Mxz*Myy - Myz*Mxy) + Myz*(Myz*Mxx - Mxz*Mxy) - Var_z*Cov_xy;
    A22 = A2 + A2;
    
    #finding the root of the characteristic polynomial
    y = A0
    x = 0.
    for i in range(IterMax):#(x=0.,y=A0,iter=0; iter<IterMAX; iter++)
        Dy = A1 + x*(A22 + 16.*x*x);
        xnew = x - y/Dy;
        if xnew == x or not np.isfinite(xnew):
            break
        ynew = A0 + xnew*(A1 + xnew*(A2 + 4.*xnew*xnew));
        if abs(ynew)>=abs(y):
            break
        x, y = xnew, ynew
        
    det = x*x - x*Mz + Cov_xy;
    Xcenter = (Mxz*(Myy - x) - Myz*Mxy)/det/2.;
    Ycenter = (Myz*(Mxx - x) - Mxz*Mxy)/det/2.;
    
    x = Xcenter + X.mean();
    y = Ycenter + Y.mean();
    r = sqrt(Xcenter*Xcenter + Ycenter*Ycenter + Mz - x - x);
    s = sigma(coords,x,y,r);
    iter_ = i;
    if verbose:
        print('Regression complete in {} iterations.'.format(iter_))
        print('Sigma computed: ', s)
    return x, y, r, s

def leastsq_circle(x,y):
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x,y))
    xc, yc = center
    Ri       = calc_R(x, y, *center)
    R        = Ri.mean()
    residu   = np.sum((Ri - R)**2)
    return xc, yc, R, residu

def plot_data_circle(x,y, xc, yc, R):
    f = plt.figure( facecolor='white')  #figsize=(7, 5.4), dpi=72,
    plt.axis('equal')

    theta_fit = np.linspace(-pi, pi, 180)

    x_fit = xc + R*np.cos(theta_fit)
    y_fit = yc + R*np.sin(theta_fit)
    plt.plot(x_fit, y_fit, 'b-' , label="fitted circle", lw=2)
    plt.plot([xc], [yc], 'bD', mec='y', mew=1)
    plt.xlabel('x')
    plt.ylabel('y')   
    # plot data
    plt.plot(x, y, 'r-.', label='data', mew=1)

    plt.legend(loc='best',labelspacing=0.1 )
    plt.grid()
    plt.title('Least Squares Circle')