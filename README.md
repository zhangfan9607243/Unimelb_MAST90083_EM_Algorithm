# Unimelb MAST90083 EM Algorithm Implementation for Image Recognition

## Acknowledgement
I would like to extend my sincere gratitude to the Unimelb MAST90083 2021S2 teaching team for providing me with the opportunity to work on this project, as well as for their guidance and feedback on my work.

## Project Introduction
In this project, we will implement the Expectation-Maximization (EM) algorithm to model the intensity distribution of an image. Specifically, we assume that the grayscale values of all pixels in the image follow a Gaussian Mixture Model (GMM) using three probability density functions (PDFs) with varying standard deviations, corresponding to the person, the camera, and the background. 

The task involves first implementing the EM algorithm for GMM and then applying it iteratively to the image data to estimate the parameters (means, standard deviations, and mixing probabilities) for each Gaussian component. Additionally, the final part of the project requires assigning each pixel in the image to one of the Gaussian distributions based on the estimated parameters and visualizing the labeled image to compare it with the provided reference output. The complete code is included in 'em_algorithm.r'.

## Prepare Data
First, we import necessary packages and load data into R.

```{r
# Packages
library(plot.matrix)
library(png)
library(fields)

# Image data
I <- readPNG("CM.png")
I <- I[, , 1]
I <- t(apply(I, 2, rev))

# Plot inmage and density of image data
par(mfrow = c(1, 2))
image(I, col = gray((0:255) / 255))
plot(density(I))
```

![3614ac73-4555-49ee-91b1-b77ecd26799f](https://github.com/user-attachments/assets/1d99d2d2-12ca-45cd-89c7-ac2818b3593e)

## EM for GMM Formulation
In case of a Gaussian Mixture Model (GGM) $Y_i$ with 3 components, we have $Y_i|Z_i=k \sim N(\mu_k,\sigma^2_k)$, where $Z_i$ takes values $1,2,3$, with probability $\pi_1, \pi_2, 1-\pi_1-\pi_2$ respectively. The parameters to be estimated by EM algorithm is $\theta = (\pi_1, \pi_2,\mu_1,\mu_2,\mu_3,\sigma^2_1,\sigma^2_2,\sigma^2_3)$.

The joint pdf of $Y_i$ and $Z_i$ is:

![CodeCogsEqn](https://github.com/user-attachments/assets/b4654c3f-ab55-4e68-a2f9-f39acf97dc5e)


The complete log likelihood function is:

![CodeCogsEqn (1)](https://github.com/user-attachments/assets/ff4af5dc-0fce-40a2-b3b2-9901c36f36e9)


In E-step of EM algorithm, the expectation of complete log likelihood function is:

![CodeCogsEqn (2)](https://github.com/user-attachments/assets/7e4cbe72-d0fb-4e77-abd3-fe96511c39b9)

In M-step of EM algorithm, we take derivative of $Q(\theta|\theta^{(m)})$ w.r.t each parameters in $\theta = (\pi_1, \pi_2,\mu_1,\mu_2,\mu_3,\sigma^2_1,\sigma^2_2,\sigma^2_3)$ to derive expressions for estimators.

In terms of $\pi_j's$, for $j=1,2$ (we have $\pi_3=1-\pi_1-\pi_2$):

![CodeCogsEqn (3)](https://github.com/user-attachments/assets/f8422a5f-77b3-42df-a329-c1063d240088)

Then, we have a equation system with 2 equations and 2 unknowns.

![CodeCogsEqn (4)](https://github.com/user-attachments/assets/8449b715-2cb0-44ba-8b05-368abdc95fda)

Solving this system give us, at (m+1)th round of iteration, for $j=1,2,3$:

![CodeCogsEqn (5)](https://github.com/user-attachments/assets/0385f238-a3c3-4511-bef6-8af649099e1f)

In terms of $\mu_j's$, for $j=1,2,3$:

![CodeCogsEqn (6)](https://github.com/user-attachments/assets/9b89a823-e1c3-4c99-b325-95f63e43f3aa)

Solving this equation gives us, at (m+1)th round of iteration, for $j=1,2,3$:

![CodeCogsEqn (7)](https://github.com/user-attachments/assets/be6e2ada-b24e-4136-ae9b-ccf9122d63d7)

In terms of $\sigma^2_j$'s, for $j=1,2,3$:

![CodeCogsEqn (8)](https://github.com/user-attachments/assets/f16b4d42-6bd3-4bc1-9204-c9004910f754)

Solving this equation gives us, at (m+1)th round of iteration, for $j=1,2,3$:

![CodeCogsEqn (9)](https://github.com/user-attachments/assets/5d97024e-26ee-4808-8e3e-8c452e40e727)

## EM for GMM Realization
Having parameter estimations, in each round of iteration, we can implement EM algorithm in R. First, we build function of EM algorithm:

```{r}
# EM algorithm function
GGM.EM <- function(X, w_init, mu_init, sigmasq_init, G) {
  # parameter X: data
  # parameter w_init: initial value of pi
  # parameter mu_init: initial value of mu
  # parameter sigmasq_init: initial value of sigmasq
  # parameter G: number of components
  
  # Stop condition: change in sum of mu's and sigma's less than e_min
  e_min <- 0.000001
  
  # List of parameters
  w_iter       <- matrix(0, nrow = 1, ncol = G)
  mu_iter      <- matrix(0, nrow = 1, ncol = G)
  sigmasq_iter <- matrix(0, nrow = 1, ncol = G)
  # Put initial values in list of parameters
  w_iter[1,]       <- w_init 
  mu_iter[1,]      <- mu_init
  sigmasq_iter[1,] <- sigmasq_init
  
  # Initiate list of log likelihood, and list of sum of mu's and sigma's
  log_liks <- c(0)
  sum_list <- c(0)
  
  # Initiate e: change in sum of mu's; b: number of iterations
  e <- 1
  b <- 1
  
  # Main iteration of EM algorithm
  while (e >= e_min) {
    
    # E-step
    # Calculate joint probability of X and Z
    probXZ <- matrix(0, nrow = length(X), ncol = G)
    for(k in 1:G) { 
      probXZ[, k] <- dnorm(X, mean = mu_iter[b,k], sd = sqrt(sigmasq_iter[b,k])) * w_iter[b,k] 
    }
    # Calculate conditional probability of Z given X
    P_ik <- probXZ / rowSums(probXZ)
    # Calculate incomplete log likelihood
    log_liks <- c(log_liks, sum(log(rowSums(probXZ))))

    # M-step 
    # Calculate estimation of parameters in each round of iteration
    w_new <- colSums(P_ik)/sum(P_ik)
    
    mu_new <- c(1:G)*0
    for (k in 1:G) { 
      mu_new[k] <- sum(P_ik[,k] * X) / sum(P_ik[,k]) 
    }
    
    sigmasq_new <- c(1:G)*0
    for (k in 1:G) { 
      sigmasq_new[k] <- sum(P_ik[,k] * ((X-mu_new[k])^2))/ sum(P_ik[,k]) 
    }
    
    # Put new values of parameters in list of parameters
    w_iter <- rbind(w_iter, w_new)
    mu_iter <- rbind(mu_iter, mu_new)
    sigmasq_iter <- rbind(sigmasq_iter, sigmasq_new)
    
    # Renew list of sum of mu's and sigma's
    sum_list <- c(sum_list, sum(c(mu_new, sqrt(sigmasq_new))))
    # Calculate change in sum of mu's and sigma's
    e <- sum_list[length(sum_list)] - sum_list[length(sum_list) - 1]
    
    # Reset b
    b <- b + 1
    
    } 
  
  # Return estimation and necessary quantities
  return(list(w_iter=w_iter, 
              mu_iter=mu_iter, 
              sigmasq_iter=sigmasq_iter, 
              log_liks=log_liks, 
              P_ik = P_ik, 
              probXZ = probXZ
              )
         )
  
}
```

## Implement EM on Data
First, we consider initial values. 

From the density plot, we see there are two clear peaks in density of data: 1st component has mean around 0.2 and second component has mean around 0.85. The third component is not clear, so we suspect that it is blended within distribution of the second component. So, we set the mean of third component to be 0.7, where there is a tiny fluctuation in density plot. So we set initial mean (0.2, 0.85, 0.7).

For variance, we can see that, the distribution of component 1 seems to be narrower so we set a lower variance for component 1. The right tail of the second peak seems to be thinner and the left tail seems to be fatter, which indicate that component with mean 0.85 has a lower variance, and component with mean 0.7 has a higher variance. In conclusion, we set initial variance (0.001, 0.001, 0.01).

For proportion, it is obvious that the component with mean 0.85 has a larger proportion than the other two. So, we set initial proportion (0.25, 0.5, 0.25).

Then, we run EM algorithm on the image data, with initial values

```{r}
# Change data into an one dimensional vector
X <- as.vector(I)
# EM algorithm on the image data
EM_estimation <- GGM.EM(X,
                        w_init=c(0.25,0.5,0.25),
                        mu_init=c(0.20,0.85,0.70),
                        sigmasq_init=c(0.001,0.001,0.01),
                        G=3
                        )

para_estimation <- round(cbind(EM_estimation$mu_iter, 
                               sqrt(EM_estimation$sigmasq_iter), 
                               EM_estimation$w_iter),
                         4)
```

Then we have a look of parameter estimation in each round of iteration

```{r}
# Parameter estimation - iteration
colnames(para_estimation) <- c("mu1","mu2","mu3","sigma1","sigma2","sigma3","pi1","pi2","pi3")
rownames(para_estimation) <- paste("Iter", 0:(nrow(para_estimation)-1))
print(para_estimation)
```

<img width="508" alt="截屏2024-10-24 08 09 26" src="https://github.com/user-attachments/assets/9b896240-bc5d-4c37-9050-8bc85692835f">


Then, we look at the final parameter estimated.

```{r}
# Parameter estimation - final
mu_final <- EM_estimation$mu_iter[nrow(EM_estimation$mu_iter),]
sigma_final <- sqrt(EM_estimation$sigmasq_iter[nrow(EM_estimation$sigmasq_iter),])
p_final <- EM_estimation$w_iter[nrow(EM_estimation$w_iter),]

# Show the parameter estimation
print(round(c(mu_final, sigma_final, p_final),4))
```

<img width="487" alt="截屏2024-10-24 08 10 24" src="https://github.com/user-attachments/assets/495a25dd-5142-440e-a866-cbca01ab4132">


Therefore, the parameter estimated $\hat{\theta} = (\hat{\pi}_1=0.2448, \hat{\pi}_2=0.5047,\hat{\pi}_3=0.2505,\hat{\mu}_1=0.2185,\hat{\mu}_2=0.8429,\hat{\mu}_3=0.7089,\hat{\sigma}^2_1=0.0572^2,\hat{\sigma}^2_2=0.0346^2,\hat{\sigma}^2_3=0.1628^2)$

Then, we classify each pixel into their distributions based on their posterior probability. That is based on maximum among $P(Z_i=k|X_i,\hat{\theta})$ for $k=1,2,3$. Also, we label each pixel based on posterior pdf multiplied by mean.

```{r
# Classify pixels into their estimated components
G <- c(1:length(X))*0
P_ik <- EM_estimation$P_ik
probXZ <- EM_estimation$probXZ
for (i in 1:length(X)) { 
  G[i] <- which.max(P_ik[i,]) 
  }

# pdf of each components multiplied by their mean
pdf <- c(1:length(X))*0
for (i in 1:length(X)) { 
  pdf[i] <- sum(probXZ[i,1:3] / sum(probXZ[i,])  * mu_final[1:3]) 
  }
```

Then we plot the labeled pixels.

```{r}
# Show the result image
image.plot(matrix(G, 398, 398))
image.plot(matrix(pdf, 398, 398))
```

![6c14b3b9-1633-414a-8f19-fb6fc9268669](https://github.com/user-attachments/assets/8a4e0735-5b57-480a-8dee-ac54021c8c56)
