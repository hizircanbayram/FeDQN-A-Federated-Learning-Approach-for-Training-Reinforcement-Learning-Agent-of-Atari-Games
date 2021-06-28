# FeDQN-A-Federated-Learning-Approach-for-Training-Reinforcement-Learning-Agent-of-Atari-Games


## **Abstract** 
> *Atari games draw a good attention for a different range of people for years. This attention has recently turned into an AI perspective for computer scientists for various reasons. Firstly, games are simple and this leads a controlled environment. Secondly, games are not that complex, thus they can be used as benchmark for a wide variety of different agents. Reinforcement learning has gained popularity in the last decade and it is commonly used in computer games to train agents. Since reinforcement learning requires a lot of data to train a good agent, requirement for data with a great variability is more important than ever nowadays. Federated learning, on the other hand, has opened great era for machine learning agents trained in different local devices via a connected system. Every device (i.e., a client) connected to a server, trains an agent with its dataset in its environments and sends its learnt weights to the server. Afterwards, the server aggregates these weights and creates global model weights with them. These weights aggregate information and transfer knowledge from different datasets collected from different environments. Such pipeline endows the global model with a generalizability power as it implicitly benefitted from the diversity of the local datasets. In this work, we, to the best of our knowledge, proposed the first federated reinforcement learning pipeline for playing atari games.* 


## **How to start?**
In order to set parameters of the pipeline, use
```python
class ARGS():
    def __init__(self):
        self.env_name = 'PongDeterministic-v4'
        self.render = False
        self.episodes = 1500
        self.batch_size = 32
        self.epsilon_start = 1.0
        self.epsilon_final=0.02
        self.seed = 1773
        
        self.use_gpu = torch.cuda.is_available()
        
        self.mode = ["rl", "fl_normal"][1]
        
        self.number_of_samples = 5 if self.mode != "rl" else 1
        self.fraction = 0.4 if self.mode != "rl" else 1
        self.local_steps = 50 if self.mode != "rl" else 100
        self.rounds = 25 if self.mode != "rl" else 25
        
        
        self.max_epsilon_steps = self.local_steps*200
        self.sync_target_net_freq = self.max_epsilon_steps // 10
        
        self.folder_name = f"runs/{self.mode}/" + time.asctime(time.gmtime()).replace(" ", "_").replace(":", "_")
        
        self.replay_buffer_fill_len = 100
```
In order to start, execute
```console
python FederatedRLPong.py
```


## **Result**
We can use federating learning with reinforcement learning without any concern. In addition to that, shown in below, when we use the early stopping technique just as the evaluation score reaches score of 21, the reinforcement learning method could be trained only 10 rounds which equals 1000 episodes; whereas, the federated learning method could be trained with 750 episodes, outperforming reinforcement learning agent. Another upside of FeDQN is, since parallel training is allowed, we have 25\% better convergence based on what episodes those agents need to be trained.

<img src="https://user-images.githubusercontent.com/23126077/123674338-2d0df100-d84a-11eb-962c-b3ef410fad67.jpg" alt="alt text" width="400" height="400">

## **Demo**
| RL w/o FL  | RL w/ FL |
| ------------- | ------------- |
| ![RL](https://user-images.githubusercontent.com/23126077/123671354-d8b54200-d846-11eb-9a5e-6dfe6f139b50.gif)              |    ![FL_RL](https://user-images.githubusercontent.com/23126077/123671408-e8348b00-d846-11eb-831d-bfe66780490f.gif)           |

    
## **Acknowledgement** 
> *This project is carried out for Artificial Intelligence Course at Istanbul Technical University during 2021 spring semester.*



