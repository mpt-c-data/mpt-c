# Multidimensional-Personality-Test-for-Children-MPT-C-
Development and Deep Learning-based Validation of a New Projective Test for Children. 

Author : Aaron Hameiri

Email : aaron.hameiri@mpt-c.org

This project contains the implementation of training Physcological variables on deep neural networks.
There are main two types of neural networks.

1. Normal Neural Networks with 162 input variables.

2. SEL and SEV Neural Networks with 25 input variables.

Note : These two special variables donot take any predefined input. The input to these variables is the predicted value from trained neural networks on 25 variables. The 25 variables are as follows.

ANHEDONIA ANXIOUSNESS ATTENTION_SEEKING CALLOUSNESS DECEITFULNESS DEPRESSIVITY DISTRACTIBILITY ECCENTRICITY EMOTIONAL_LABILITY GRANDIOSITY HOSTILITY IMPULSIVITY INTIMACY_AVOIDANCE IRRESPONSIBILITY MANIPULATIVENESS PERCEPTUAL_DYSREGULATION PERSEVERATION RESTRICTED_AFFECTIVITY RIGID_PERFECTIONISM RISK_TAKING SEPARATION_INSECURITY SUBMISSIVENESS SUSPICIOUSNESS UNUSUAL_BELIEFS WITHDRAWAL.

The procdure is to train these Neural Networks of 25 variables on normal input. Than use those 25 neural networks to get a input of 25 variables for the training of the special variables i.e. SEL and SEV 
