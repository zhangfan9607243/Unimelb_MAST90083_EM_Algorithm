# Packages
library(plot.matrix)
library(png)
library(fields)
# Image data
I <- readPNG("CM.png")
I <- I [, , 1]
I <- t(apply(I, 2, rev))
# Plot inmage and density of image data
par(mfrow = c(1, 2))
image(I, col = gray((0:255) / 255))
plot(density(I))

# EM algorithm function
GGM.EM = function(X, w_init, mu_init, sigmasq_init, G) {
  # parameter X: data
  # parameter w_init: initial value of pi
  # parameter mu_init: initial value of mu
  # parameter sigmasq_init: initial value of sigmasq
  # parameter G: number of components
  
  # Stop condition: change in sum of mu's and sigma's less than e_min
  e_min <- 0.000001
  
  # List of parameters
  w.iter       <- matrix(0, nrow = 1, ncol = G)
  mu.iter      <- matrix(0, nrow = 1, ncol = G)
  sigmasq.iter <- matrix(0, nrow = 1, ncol = G)
  # Put initial values in list of parameters
  w.iter[1,]       <- w_init 
  mu.iter[1,]      <- mu_init
  sigmasq.iter[1,] <- sigmasq_init
  
  # Initiate list of log likelihood, and list of sum of mu's and sigma's
  log_liks = c(0)
  sum_list = c(0)
  
  # Initiate e: change in sum of mu's; b: number of iterations
  e = 1
  b = 1
  
  # Main iteration of EM algorithm
  while (e >= e_min) {
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
  return(list(w.iter=w.iter, mu.iter=mu.iter, sigmasq.iter=sigmasq.iter, 
              log_liks=log_liks, P_ik = P_ik, probXZ = probXZ))
  
}
# Change data into an one dimensional vector
X = as.vector(I)
# EM algorithm on the image data
EM_estimation = GGM.EM(X,w_init=c(0.25,0.5,0.25),mu_init=c(0.20,0.85,0.70),
                       sigmasq_init=c(0.001,0.001,0.01),G=3)

para_estimation = round(cbind(EM_estimation$mu.iter, 
                              sqrt(EM_estimation$sigmasq.iter), 
                              EM_estimation$w.iter),4)
colnames(para_estimation) = c("mu1","mu2","mu3","sigma1","sigma2","sigma3","pi1","pi2","pi3")
rownames(para_estimation) = paste("Iter", 0:(nrow(para_estimation)-1))
print(para_estimation)

mu_final = EM_estimation$mu.iter[nrow(EM_estimation$mu.iter),]
sigma_final = sqrt(EM_estimation$sigmasq.iter[nrow(EM_estimation$sigmasq.iter),])
p_final = EM_estimation$w.iter[nrow(EM_estimation$w.iter),]
# Show the parameter estimation
print(round(c(mu_final, sigma_final, p_final),4))

G = c(1:length(X))*0
P_ik = EM_estimation$P_ik
probXZ = EM_estimation$probXZ
for (i in 1:length(X)) { G[i] = which.max(P_ik[i,]) }
# pdf of each components multiplied by their mean
pdf = c(1:length(X))*0

for (i in 1:length(X)) { 
    pdf[i] = sum(probXZ[i,1:3] / sum(probXZ[i,])  * mu_final[1:3]) 
    }

image.plot(matrix(G, 398, 398))
image.plot(matrix(pdf, 398, 398))

