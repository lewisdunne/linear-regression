''' Example of linear regression solution '''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate a fake dataset
data_n = 1000
data_mean = 0
data_stdv = 1
X = np.arange(1, data_n+1, 1)
y = np.random.normal(data_mean, data_stdv, data_n)
# Update y to give it a trend
y = np.array([yi + Xi/500 for yi, Xi in zip(y, X)])

#-Functions ------------------------------------------------------------------#
def linear_regression(X, y):
    """
    Linear regression function:
        Solves for β0 and β1
    """
    # Calculate means
    Xbar, ybar = np.mean(X), np.mean(y) # All regression lines pass through Xbar, ybar
    
    # Solve for the slope (beta1)
    X_err = X-Xbar # X errors
    y_err = y-ybar # y errors
    err_prod = X_err * y_err # product of X and y errors. This is the numerator
    SSE_X = np.sum(X_err**2) # Sum of squared errors for X. This is the denominator
    beta1 = sum(err_prod) / SSE_X # our beta1 coefficient (slope)
    
    # Solve for the intercept (beta0)
    # We know that y=ybar when X=Xbar, since all lines pass this intersection.
    # Therefore beta0 is the difference between ybar and beta1.
    beta0 = ybar - (beta1*Xbar)
    
    return beta0, beta1

def predict_y(x, beta0, beta1):
    """
    Use params beta0 and beta1 to predict a hypothetical val of x.
    """
    y = beta0 + beta1 * x
    return y

#-Testing: Calculating -------------------------------------------------------#
# Call the func
b0, b1 = linear_regression(X, y)

# Make line
max_pred = max(X)
min_pred = min(X)
linex = np.arange(min_pred, max_pred, 1) # x values
liney = [predict_y(x, b0, b1) for x in linex]

#-Testing: Plotting ----------------------------------------------------------#
fig, ax = plt.subplots(1,2, figsize=(15, 8))

ax[0].axvline(x=np.mean(X), lw=0.75, ls='dashed', c='gray', zorder=0)
ax[0].axhline(y=np.mean(y), lw=0.75, ls='dashed', c='gray', zorder=0)

ax[0].scatter(X, y, s=5, c='k')
ax[0].plot(linex, liney, c='r', lw=2)
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_title('Manual Linear Reg.')

# Check it against seaborn's regplot, which calculates regression automatically
ax[1].axvline(x=np.mean(X), lw=0.75, ls='dashed', c='gray', zorder=0)
ax[1].axhline(y=np.mean(y), lw=0.75, ls='dashed', c='gray', zorder=0)
sns.regplot(X, y, ax=ax[1])

ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_title('Seaborn\'s Linear Reg.')

plt.suptitle('Linear Regression Example')
plt.tight_layout()
plt.show()
#-----------------------------------------------------------------------------#
