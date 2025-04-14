# NeuralVelocity
NeVe - Neural Velocity for hyperparameter tuning (IJCNN 2025)

<b>Abstract</b>:

Hyperparameter tuning, such as learning rate decay and defining a stopping criterion, often relies on monitoring the validation loss. 

This paper presents NeVe, a dynamic training approach that adjusts the learning rate and defines the stop criterion based on the novel notion of “neural velocity”. 
The neural velocity measures the rate of change of each neuron’s transfer function and is an indicator of model convergence: sampling neural velocity can be performed even by forwarding noise in the network, reducing the need for a held-out dataset. 

Our findings show the potential of neural velocity as a key metric for optimizing neural network training efficiently

![PDF Teaser](assets/teaser.pdf)

![Teaser](assets/teaser.png)
