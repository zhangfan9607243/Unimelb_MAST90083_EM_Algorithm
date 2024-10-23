<script type="text/javascript" src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Unimelb MAST90083 EM Algorithm Implementation for Image Recognition

First, we import necessary packages and load data into R.

```{r
# Packages
library(plot.matrix) 
library(png)
library(fields)

# Image data
I = readPNG("CM.png")
I = I [,,1]
I = t(apply(I,2,rev))

# Plot inmage and density of image data
par(mfrow = c(1,2))
image(I, col = gray((0:255)/255))
plot(density(I))
```

![3614ac73-4555-49ee-91b1-b77ecd26799f](https://github.com/user-attachments/assets/1d99d2d2-12ca-45cd-89c7-ac2818b3593e)

In case of a Gaussian Mixture Model (GGM) $Y_i$ with 3 components, we have $Y_i|Z_i=k \sim N(\mu_k,\sigma^2_k)$, where $Z_i$ takes values $1,2,3$, with probability $\pi_1, \pi_2, 1-\pi_1-\pi_2$ respectively. The parameters to be estimated by EM algorithm is $\theta = (\pi_1, \pi_2,\mu_1,\mu_2,\mu_3,\sigma^2_1,\sigma^2_2,\sigma^2_3)$.

The joint pdf of $Y_i$ and $Z_i$ is:

$$
\begin{aligned}

f(y_i,z_i|\theta)&=f(y_i|z_i,\theta)f(z_i|\theta)\\

&=\prod_{k=1}^{3}[f(y_i|z_i=k,\theta)f(z_i=k|\theta)]^{I(Z_i=k)}\\

&=\prod_{k=1}^{3}[f(y_i|z_i=k,\theta)\pi_k]^{I(Z_i=k)}\\

\end{aligned}
$$

The complete log likelihood function is:

$$
\begin{aligned}

log[f(Y,Z|\theta)]&=log[\prod_{i=1}^{n} f(y_i,z_i|\theta)]\\

&=\sum_{i=1}^{n} log[f(y_i,z_i|\theta)]\\

&=\sum_{i=1}^{n}\sum_{k=1}^{3}I(Z_i=k)(log(\pi_k)+log(f(y_i|z_i=k,\theta)))\\

\end{aligned}
$$

In E-step of EM algorithm, the expectation of complete log likelihood function is:

$$
\begin{aligned}

Q(\theta|\theta^{(m)})&=E(log[f(Y,Z|\theta)]|Y,\theta)\\

&=E(\sum_{i=1}^{n}\sum_{k=1}^{3}I(Z_i=k)(log(\pi_k)+log(f(y_i|z_i=k,\theta)))|Y,\theta^{(m)})\\

&=\sum_{i=1}^{n}\sum_{k=1}^{3}E(I(Z_i=k)|Y,\theta^{(m)})(log(\pi_k)+log(f(y_i|z_i=k,\theta)))\\

&=\sum_{i=1}^{n}\sum_{k=1}^{3}P(Z_i=k|Y,\theta^{(m)})(log(\pi_k)+log(f(y_i|z_i=k,\theta)))\\

\end{aligned}
$$

In M-step of EM algorithm, we take derivative of $Q(\theta|\theta^{(m)})$ w.r.t each parameters in $\theta = (\pi_1, \pi_2,\mu_1,\mu_2,\mu_3,\sigma^2_1,\sigma^2_2,\sigma^2_3)$ to derive expressions for estimators.

In terms of $\pi_j's$, for $j=1,2$ (we have $\pi_3=1-\pi_1-\pi_2$):

$$
\begin{aligned}

\frac{\partial Q}{\partial\pi_j}&=\frac{\partial}{\partial\pi_j}\sum_{i=1}^{n}\sum_{k=1}^{3}P(Z_i=k|Y,\theta^{(m)})(log(\pi_k)+log(f(y_i|z_i=k,\theta)))\\

&=\frac{\partial}{\partial\pi_j}\sum_{i=1}^{n}[P(Z_i=j|Y,\theta^{(m)})log(\pi_j)+P(Z_i=3|Y,\theta^{(m)})log(\pi_3)]\\

&=\sum_{i=1}^{n}[\frac{P(Z_i=j|Y,\theta^{(m)})}{\pi_j}-\frac{P(Z_i=3|Y,\theta^{(m)})}{\pi_3}]=0\\

\Longrightarrow& \pi_3\sum_{i=1}^{n}P(Z_i=j|Y,\theta^{(m)})=\pi_j\sum_{i=1}^{n}P(Z_i=3|Y,\theta^{(m)})

\end{aligned}
$$

Then, we have a equation system with 2 equations and 2 unknowns.

$$
\begin{cases}\pi_3\sum_{i=1}^{n}P(Z_i=1|Y,\theta^{(m)})=\pi_1\sum_{i=1}^{n}P(Z_i=3|Y,\theta^{(m)})\\ \pi_3\sum_{i=1}^{n}P(Z_i=2|Y,\theta^{(m)})=\pi_2\sum_{i=1}^{n}P(Z_i=3|Y,\theta^{(m)})\\ 

\pi_3=1-\pi_1-\pi_2\\

\end{cases}
$$

Solving this system give us, at (m+1)th round of iteration, for $j=1,2,3$:

$$
\hat{\pi}^{(m+1)}_k=\frac{1}{n}\sum_{i=1}^{n} P(Z_i=k|Y,\theta^{(m)})
$$

In terms of $\mu_j's$, for $j=1,2,3$:

$$
\begin{aligned}

\frac{\partial Q}{\partial\mu_j}&=\frac{\partial}{\partial\mu_j}\sum_{i=1}^{n}\sum_{k=1}^{3}P(Z_i=k|Y,\theta^{(m)})(log(\pi_k)+log(f(y_i|z_i=k,\theta)))\\

&= \frac{\partial}{\partial\mu_j}\sum_{i=1}^{n} P(Z_i=j|Y,\theta^{(m)})(log(\pi_j)+log(\frac{1}{\sqrt{2\pi\sigma^2_j}}exp(-\frac{(y_i-\mu_j)^2}{2\sigma^2_j})))\\

&=\sum_{i=1}^{n} P(Z_i=j|Y,\theta^{(m)})\frac{\partial}{\partial\mu_j}(-\frac{(y_i-\mu_j)^2}{2\sigma^2_j})\\

&=\sum_{i=1}^{n} P(Z_i=j|Y,\theta^{(m)})(-\frac{1}{2\sigma^2_j})(-2(y_i-\mu_j))\\

&= (\frac{1}{\sigma^2_j})[\sum_{i=1}^{n} P(Z_i=j|Y,\theta^{(m)})y_i-\mu_j\sum_{i=1}^{n} P(Z_i=j|Y,\theta^{(m)})]=0\\

\end{aligned}
$$

Solving this equation gives us, at (m+1)th round of iteration, for $j=1,2,3$:

$$
\hat{\mu}^{(m+1)}_k=\frac{\sum_{i=1}^{n} P(Z_i=k|Y,\theta^{(m)})y_i}{\sum_{i=1}^{n} P(Z_i=k|Y,\theta^{(m)})}
$$

In terms of $\sigma^2_j$'s, for $j=1,2,3$:

$$
\begin{aligned}

\frac{\partial Q}{\partial\sigma^2_j}&=\frac{\partial}{\partial\sigma^2_j}\sum_{i=1}^{n}\sum_{k=1}^{3}P(Z_i=k|Y,\theta^{(m)})(log(\pi_k)+log(f(y_i|z_i=k,\theta)))\\

&= \frac{\partial}{\partial\sigma^2_j}\sum_{i=1}^{n} P(Z_i=j|Y,\theta^{(m)})(log(\pi_j)+log(\frac{1}{\sqrt{2\pi\sigma^2_j}}exp(-\frac{(y_i-\mu_j)^2}{2\sigma^2_j})))\\

&= \sum_{i=1}^{n} P(Z_i=j|Y,\theta^{(m)})\frac{\partial}{\partial\sigma^2_j}(-\frac{1}{2}log(2\pi\sigma^2_j)-\frac{(y_i-\mu_j)^2}{2\sigma^2_j})\\

&=\sum_{i=1}^{n} P(Z_i=j|Y,\theta^{(m)})(-\frac{1}{2\sigma^2_j}+\frac{(y_i-\mu_j)^2}{2(\sigma^2_j)^2})=0\\

\end{aligned}
$$

Solving this equation gives us, at (m+1)th round of iteration, for $j=1,2,3$:

$$
\hat{\sigma}^{2(m+1)}_k=\frac{\sum_{i=1}^{n} P(Z_i=k|Y,\theta^{(m)})(y_i-\hat{\mu}^{(m+1)}_k)^2}{\sum_{i=1}^{n} P(Z_i=k|Y,\theta^{(m)})}
$$

Having parameter estimations, in each round of iteration, we can implement EM algorithm in R. First, we build function of EM algorithm:

```{r}
# EM algorithm function
GGM.EM = function(X, w.init, mu.init, sigmasq.init, G) {
  # parameter X: data
  # parameter w.init: initial value of pi
  # parameter mu.init: initial value of mu
  # parameter sigmasq.init: initial value of sigmasq
  # parameter G: number of components

  # Stop condition: change in sum of mu's and sigma's less than e.min
  e.min = 0.000001

  # List of parameters
  w.iter       = matrix(0, nrow = 1, ncol = G)
  mu.iter      = matrix(0, nrow = 1, ncol = G)
  sigmasq.iter = matrix(0, nrow = 1, ncol = G)

  # Put initial values in list of parameters
  w.iter[1,]       = w.init 
  mu.iter[1,]      = mu.init
  sigmasq.iter[1,] = sigmasq.init

  # Initiate list of log likelihood, and list of sum of mu's and sigma's
  log_liks = c(0)
  sum_list = c(0)

  # Initiate e: change in sum of mu's; b: number of iterations
  e = 1
  b = 1

  # Main iteration of EM algorithm
  while (e >= e.min) {
  
    # E-step

    # Calculate joint probability of X and Z
    probXZ = matrix(0, nrow = length(X), ncol = G)
    for(k in 1:G) { 
      probXZ[, k] = dnorm(X, mean = mu.iter[b,k], sd = sqrt(sigmasq.iter[b,k])) * w.iter[b,k] 
    }

    # Calculate conditional probability of Z given X
    P_ik = probXZ / rowSums(probXZ)

    # Calculate incomplete log likelihood
    log_liks = c(log_liks, sum(log(rowSums(probXZ))))

    # M-step 

    # Calculate estimation of parameters in each round of iteration
    w.new = colSums(P_ik)/sum(P_ik)
    mu.new = c(1:G)*0
    for (k in 1:G) { 
      mu.new[k] = sum(P_ik[,k] * X) / sum(P_ik[,k]) 
    }

    sigmasq.new = c(1:G)*0
    for (k in 1:G) { 
      sigmasq.new[k] = sum(P_ik[,k] * ((X-mu.new[k])^2))/ sum(P_ik[,k]) 
    }

    # Put new values of parameters in list of parameters
    w.iter = rbind(w.iter, w.new)
    mu.iter = rbind(mu.iter, mu.new)
    sigmasq.iter = rbind(sigmasq.iter, sigmasq.new)

    # Renew list of sum of mu's and sigma's
    sum_list = c(sum_list, sum(c(mu.new, sqrt(sigmasq.new))))

    # Calculate change in sum of mu's and sigma's
    e = sum_list[length(sum_list)] - sum_list[length(sum_list) - 1]

    # Reset b
    b = b + 1

    } 

  # Return estimation and necessary quantities
  return(list(w.iter=w.iter, 
	      mu.iter=mu.iter, 
	      sigmasq.iter=sigmasq.iter, 
              log_liks=log_liks, 
	      P_ik = P_ik, 
	      probXZ = probXZ))

}

```

Then, we consider initial values.

From the density plot, we see there are two clear peaks in density of data: 1st component has mean around 0.2 and second component has mean around 0.85. The third component is not clear, so we suspect that it is blended within distribution of the second component. So, we set the mean of third component to be 0.7, where there is a tiny fluctuation in density plot. So we set initial mean (0.2, 0.85, 0.7).

For variance, we can see that, the distribution of component 1 seems to be narrower so we set a lower variance for component 1. The right tail of the second peak seems to be thinner and the left tail seems to be fatter, which indicate that component with mean 0.85 has a lower variance, and component with mean 0.7 has a higher variance. In conclusion, we set initial variance (0.001, 0.001, 0.01).

For proportion, it is obvious that the component with mean 0.85 has a larger proportion than the other two. So, we set initial proportion (0.25, 0.5, 0.25).

Then, we run EM algorithm on the image data, with initial values

```{r}
# Change data into an one dimensional vector
X = as.vector(I)

# EM algorithm on the image data
EM_estimation = GGM.EM(X,
		       w.init=c(0.25,0.5,0.25),
		       mu.init=c(0.20,0.85,0.70),
                       sigmasq.init=c(0.001,0.001,0.01),
		       G=3)

```

Then we have a look of parameter estimation in each round of iteration

```{r}
para_estimation = round(cbind(EM_estimation$mu.iter, 
                              sqrt(EM_estimation$sigmasq.iter), 
                              EM_estimation$w.iter),4)

colnames(para_estimation) = c("mu1","mu2","mu3","sigma1","sigma2","sigma3","pi1","pi2","pi3")
rownames(para_estimation) = paste("Iter", 0:(nrow(para_estimation)-1))
print(para_estimation)
```

Then, we look at the final parameter estimated.

```{r}
# Parameter estimation
mu_final = EM_estimation$mu.iter[nrow(EM_estimation$mu.iter),]
sigma_final = sqrt(EM_estimation$sigmasq.iter[nrow(EM_estimation$sigmasq.iter),])
p_final = EM_estimation$w.iter[nrow(EM_estimation$w.iter),]

# Show the parameter estimation
print(round(c(mu_final, sigma_final, p_final),4))
```

Therefore, the parameter estimated $\hat{\theta} = (\hat{\pi}_1=0.2448, \hat{\pi}_2=0.5047,\hat{\pi}_3=0.2505,\hat{\mu}_1=0.2185,\hat{\mu}_2=0.8429,\hat{\mu}_3=0.7089,\hat{\sigma}^2_1=0.0572^2,\hat{\sigma}^2_2=0.0346^2,\hat{\sigma}^2_3=0.1628^2)$

Then, we classify each pixel into their distributions based on their posterior probability. That is based on maximum among $P(Z_i=k|X_i,\hat{\theta})$ for $k=1,2,3$. Also, we label each pixel based on posterior pdf multiplied by mean.

```{r
# Classify pixels into their estimated components
G = c(1:length(X))*0
P_ik = EM_estimation$P_ik
probXZ = EM_estimation$probXZ
for (i in 1:length(X)) { G[i] = which.max(P_ik[i,]) }

# pdf of each components multiplied by their mean
pdf = c(1:length(X))*0
for (i in 1:length(X)) { pdf[i] = sum(probXZ[i,1:3] / sum(probXZ[i,])  * mu_final[1:3]) }
```

Then we plot the labeled pixels.

```{r}
# Show the result image
image.plot(matrix(G, 398, 398))
image.plot(matrix(pdf, 398, 398))
```
