# Multidimensional-Personality-Test-for-Children-MPT-C-
Development and Deep Learning-based Validation of a New Projective Test for Children. 

Author : Aaron Hameiri

Email : aaron.hameiri@mpt-c.org

This project contains the implementation of training Physcological variables on deep neural networks.
There are main two types of neural networks.

1. Normal Neural Networks with 162 input variables.

2. SEL and SEV Neural Networks with 25 input variables.

Note : These two special variables donot take any predefined input. The input to these variables is the predicted value from trained neural networks on 25 variables. The 25 variables are as follows.

1.ANHEDONIA 

2.ANXIOUSNESS 

3.ATTENTION_SEEKING 

4.CALLOUSNESS 

5.DECEITFULNESS 

6.DEPRESSIVITY 

7.DISTRACTIBILITY 

8.ECCENTRICITY 

9.EMOTIONAL_LABILITY 

10.GRANDIOSITY 

11.HOSTILITY 

12.IMPULSIVITY 

13.INTIMACY_AVOIDANCE 

14.IRRESPONSIBILITY 

15.MANIPULATIVENESS 

16.PERCEPTUAL_DYSREGULATION 

17.PERSEVERATION 

18.RESTRICTED_AFFECTIVITY 

19.RIGID_PERFECTIONISM 

20.RISK_TAKING 

21.SEPARATION_INSECURITY 

22.SUBMISSIVENESS 

23.SUSPICIOUSNESS 

24.UNUSUAL_BELIEFS 

25.WITHDRAWAL

The procdure is to train these Neural Networks of 25 variables on normal input. Than use those 25 neural networks to get a input of 25 variables for the training of the special variables i.e. SEL and SEV 



# Gradient Explosion

Due to very difficult dataset. Gradient explosion is very common. While training the neural Network the user may notice NAN in the loss. 

