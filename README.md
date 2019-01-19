# Learning Multi-Agent Communication in Reinforcement Learning
======
## Final project of Reinforcement Learning (MVA 2018-2019)
------

*F. Gaschi, R. Zimmer*


For this project, we implemented a simple version of CommNet, a flexible controler structure for multi-agent communication learning. This architecture is proposed by S. Sukhbaatar, A. Szlam,and R. Fergus. Learning multiagent communication with backpropagation. NIPS, 2016. (http://papers.nips.cc/paper/6397-learning-multiagent-communication-with-backpropagation)

The CommNetLever model is used for a simple lever pulling task proposed by S. Sukhbaatar, A. Szlam,and R. Fergus in their original paper. This model can be trained with supervised learning or REINFORCE algorithm.

The CommNetPP (Predator-Prey) model is a slightly modified version of the original CommNet architecture with GRU modules where agents can decide whether or not they want to communicate. This model is trained with REINFORCE algorithm and is used to solve a Predator-Prey task.

