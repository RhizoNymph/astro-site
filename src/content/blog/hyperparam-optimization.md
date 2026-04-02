---
title: Introduction to Hyperparameter Optimization for Neural Nets
pubDate: 2025-08-18
---
# Introduction to Hyperparamter Optimization for Neural Nets
<br>
I recently began training neural networks again for the first time since Tensorflow was the New Hotness, this time with pytorch. I won’t talk much about pytorch here, that’s for a different article. What I will talk about here is something that will be more generally applicable, hyperparameters. These are basically a bunch of knobs you can turn in an effort to make the training process go more smoothly, but it’s difficult to know how to set these. This isn’t a problem specific to neural nets, I often found myself performing similar optimization processes in my days training random forest models on time series data. The particular hyperparameters that I chose were different because I was using neural nets, but the idea of hyperparameter optimization is applicable to things as simple as a least squares regression with regularization. 


<br>To give some context, the project I’m working on at the moment is improving agentic search agents through improving query understanding so the LLM can craft more effective search queries. It’s difficult to actually craft a rules based reward function to do this though, so I decided to train a reward model as the first step to this. I didn’t want to start from scratch though, so I chose to use transfer learning with the ModernBERT model from [Answer.ai](http://Answer.ai) as an encoder in an effort to cut down the amount of data and training time that I would need. Not every training process will look like this one, particularly with respect to the model parameters where I talk about the encoder and regression head. 

<br>It’s important to me to make sure that I (and by extension you) understand exactly what each of these knobs does. For these explanations to really hit, you (and I) have to understand some basics of neural nets and gradient descent (the process by which neural nets learn). I don’t want to make this article into neural nets 101, so I’m going to try to keep it short. If you already have a background in these topics, feel free to skip the next section.

<br>

### **Neural Nets & Gradient Descent**

<br>Neural nets learn to map inputs to outputs using successive layers of weights and biases, interspersed with activation functions. The input data gets multiplied by the weights, and biases shift those outputs before they are put through activation functions. Activation functions add nonlinearities to the network so that it can learn to model nonlinear relationships, otherwise it would collapse into a linear transformation. This process can be successively applied many times, as well as with regularization techniques (which attempt to not overfit the training data) applied throughout.


<br>The weights and biases are set using gradient descent, which uses an initial (often random) guess for the value of the parameter being estimated to get a loss value for that guess, calculating the derivative of the function that models the loss value with respect to that parameter. When there are multiple parameters organized in a matrix, the set of partial derivatives is called a gradient. After calculating the gradient, the algorithm knows which way to adjust parameters to seek the minimum of the loss function. It takes a step in that direction, scaled by a hyperparameter called the learning rate in order to not overshoot the minimum. This is formulated by the following equation:


<br>t+1= t-η∇L(t)


<br>Where t is the parameter set at timestep t, η is the learning rate, and ∇L(t) is the gradient of L with respect to  of the set of t.


<br>However, calculating the gradient is computationally intensive because it involves all of the pieces of data from the dataset so to save time we use batches of a given size randomly sampled (without replacement) from the dataset. This increases the variance of gradient estimates by introducing noise into the training process, but speeds up the optimization process dramatically and lowers the memory burden of training enough that it’s a ubiquitous technique.

<br>

### **Hyperparameters**

<br>Model architecture parameters:


- Dropout is a technique to avoid overfitting where a random subset of the activations are set to 0 during each iteration at the point where dropout is included. This parameter controls the fraction of activations that are affected. In this network it is applied to the pooled output of the pretrained ModernBERT encoder and after each hidden layer, but you can also include the placement of dropout as another hyperparameter. It’s important to note that dropout is skipped during post training inference.  
- Num\_hidden\_layers controls the depth of the network after the encoder, applying the combination of linear layer, layernorm (if used), activation function, and dropout the number of times that this parameter is set to.  
- Hidden\_dim\_ratio controls the ratio at which hidden layers shrink the dimensions of the encoder portion of the network and each successive layer for dimensionality reduction before the output layer.  
- Activation controls the activation function used in between the linear layers and the dropout layers, here it chooses from relu, gelu, tanh, and leaky\_relu. This can vary the way that the network models complex, nonlinear relationships so the optimal function to use is dependent on the data itself.  
- Use\_layer\_norm controls whether or not the hidden layers perform a layernorm operation in between the linear layer and activation function. Layernorm normalizes the activations of the layer it’s applied to in an effort to stabilize gradient calculations.


<br>Training parameters:

- Learning\_rate controls the base learning rate of the training process that is then used to calculate the learning rate for the regression head and fed into the linear scheduler to determine the rates over the entire process. Learning rate refers to the step size of the gradient descent process when attempting to step towards the minimum of the loss landscape.  
- Head\_lr\_multiplier controls the multiplier for the base learning rate when calculating the rate used for the regression head. It is sometimes applicable that the optimal learning rate for different parts of a network can be different, so this is a common parameter to allow a different rate for the head of the network.  
- Batch\_size controls the number of samples used in each batch during gradient descent. Smaller values decrease memory usage and can speed up the process of training but also introduce more noise to your gradient estimates.  
- Weight\_decay controls the degree to which large weights are penalized. This is used to increase generalization to data outside the training set, as large values can cause the network to be very sensitive to small changes in inputs which results in being particularly tuned to the training data.  
- Warmup ratio refers to the portion of total training steps across which the learning rate is scaled up from 0 to reach the initial value. This is useful because the initial random estimates of parameters and gradients can have scales that differ greatly from what they will be as the process stabilizes, so having a smaller learning rate in the beginning allows these estimates to have time to ‘settle’ before increasing it and allowing them to move around in proper scales.  
- Gradient\_accumulation\_steps controls how many batches are used to calculate gradients before updating the weights of the model, often used in situations where memory is limited to simulate a higher batch size.  
- Max\_grad\_norm controls the maximum value that gradients are scaled to if the gradient norm exceeds it. This is a process that is applied after each gradient calculation in an effort to prevent the gradients from ‘vanishing’ or ‘exploding’. This is a problem that is particularly relevant in deep networks, as layers tend to cause gradients to shrink or grow and many successive layers can reduce them to 0 or increase them to very large numbers which causes instability in choosing step sizes towards minimum losses.   
- Max\_length controls the maximum sequence length to which input sequences are padded or truncated to be equivalent to. Training at larger sequence lengths than your dataset or expected inputs after training increases required compute and memory requirements while also potentially decreasing generalization to out of distribution data.  
- Num\_epochs controls the number of times the entire dataset is put through the training loop. If the dataset consists of 1000 samples, the batch\_size is 10, and num\_epochs is 2 then the total training steps in the process is 200\. Total steps \= Number of samples /  batch size \* number of epochs  
- The learning rate is often varied across the training process using a scheduler. You can use a constant rate, but in practice it’s more common to decay your learning rate linearly and most common to decay it using a ‘cosine’ schedule. Whether or not to use a scheduler and which type can be considered to be hyperparameters. If you want to narrow your search space though, use a cosine scheduler because the largest changes in the training process tend to happen early on, meaning you can use smaller learning rates towards the end and approach them faster than a linear schedule would.


<br>Optimizer parameters:

- The optimizer itself can be a hyperparameter, but this usually affects how the training process progresses more than it does the end result. For instance, using Adam/AdamW will usually converge faster than Stochastic Gradient Descent. Sometimes it can affect the generalization due to changing how the loss landscape is explored, but often the optimizer itself is chosen based on domain knowledge of what works best for the problem you are modeling. Optimizers themselves can have hyperparameters that are worth tuning though, as once you have chosen an optimizer those do have an effect on how the loss landscape is explored to find minima.  
- For Adam/AdamW (the most common in NLP tasks), the primary parameters to optimize are 1 and 2, which are the decay rates of estimates for the mean and variance of gradient estimates. This optimizer family uses the mean as a sort of momentum term and variance as an adaptive learning rate that scales step size based on how noisy local gradients are. The combination allows moving quickly across flat parts of the loss landscape and slowly across steep parts.

<br>

### **Hyperparameter Optimization**

<br>Phew, so that was a lot. Understanding what these hyperparameters are is a lot to take in, but the optimization process isn’t quite so complicated. You just use a subset of the different hyperparameter configurations you can possibly use, train on your validation set with each one, then take the one that had the lowest validation loss, and then use that configuration with your full training set and evaluate on your test set. It definitely can be more complex if you want to understand all of the different samplers you can use to explore your search space, but I’m going to focus on just the most basic few. To start with, you need to choose a library for the optimizer. I use Optuna, but you can also use hyperopt and even wandb (the Weights and Biases library) has included sweeps recently. I’m used to Optuna and I find it to be pretty simple so it’s what I’m using for now. I do plan to give wandb a try later because they have really nice visualizations.


<br>After you’ve chosen a library and defined your set of hyperparameters to optimize over, you need to choose a sampler. The most intuitive is grid search, which systematically explores the entire search space by using every configuration in a grid defined by minimums, maximums, and step sizes for each hyperparameter. It’s extremely computationally intensive though, so it’s rare to see this used in production use cases where you’re optimizing against a large dataset and hyperparameter search space. The logical next step is a random sampler, which just uses random samples of each hyperparameter across the allowed range. It may not find the most optimal configuration but it usually finds ones that are fairly close MUCH faster than grid search.


<br>In practice, it’s best to use a bayesian optimizer because it explores your search space much more efficiently than grid search and more exhaustively than random search. These samplers use bayesian statistics by treating the loss as a random function of the hyperparameters, taking a sample of the available space, and using the resulting losses to form posterior probabilities to predict the mean and standard deviation of the loss at each hyperparameter value, using these to choose the next set of values to use, repeating the process to narrow down the search space while intentionally exploring high uncertainty areas of the search space to improve the estimates within each iteration. Gaussian process based sampling is the most basic form of this, but it tends to struggle with parallelization, contingent probabilities, discrete variables, and high dimensionality. For these reasons the default sampler in Optuna is the TPESampler which is based on Tree-structured Parzen Estimation. This works by splitting observations into a good set and bad set, building separate density models for each hyperparameter, and scoring each candidate configuration by the ratio of expected probability of being in the good or bad set. It favors the ones where they are much more likely to be in the good set than the bad set, then evaluates them and updates the density estimates and iterates with this process.


<br>The last thing I want to cover is how to actually do this in code, because conceptual understanding only takes you so far. You start by defining a class BayesianOptimizer, which will have 3 functions of \_\_init\_\_, objective, and optimize. My init function looks like this:

<br>

```
class BayesianOptimizer:
   def __init__(
       self,
       train_queries: list[str],
       train_scores: list[float],
       val_queries: list[str],
       val_scores: list[float],
       tokenizer,
       device: torch.device,
       n_trials: int = 20,
       timeout: int | None = None,
       model_id: str = "answerdotai/ModernBERT-base"
   ):
       self.train_queries = train_queries
       self.train_scores = train_scores
       self.val_queries = val_queries
       self.val_scores = val_scores
       self.tokenizer = tokenizer
       self.device = {'device': device}
       self.n_trials = n_trials
       self.timeout = timeout
       self.model_id = model_id
```


<br>Which I feel is pretty self explanatory at this point. Train and val queries and scores are just your features and targets, the tokenizer is used to convert the natural language queries into numbers so the neural net can model them correctly, device is usually ‘cuda’ or ‘cpu’ (it should be cuda, lets be real), the number of trials is how many configurations you want to test, timeout is a maximum time for the trials to take overall, and model\_id is the pretrained model you want to use for the encoder (this part won’t be relevant unless you’re doing transfer learning).


<br>The objective function should define your hyperparameter dictionary, set up your datasets and dataloaders, set up any other inputs your model needs, create the model, send the model and dataloaders to your trainer, and train. The output should be the validation loss you get for the configuration chosen for the trial. For performance reasons you should clear your cuda cache and delete the model and trainer from memory afterwards. Here’s mine:

<br>

```
def objective(self, trial: Trial) -> float:
       hp_dict = {
           'dropout': trial.suggest_float('dropout', 0.0, 0.5, step=0.05),
           'num_hidden_layers': trial.suggest_int('num_hidden_layers', 1, 3),
           'hidden_dim_ratio': trial.suggest_float('hidden_dim_ratio', 0.25, 0.75, step=0.25),
           'activation': trial.suggest_categorical('activation', ['relu', 'gelu', 'tanh', 'leaky_relu']),
           'use_layer_norm': trial.suggest_categorical('use_layer_norm', [True, False]),


           'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True),
           'head_lr_multiplier': trial.suggest_float('head_lr_multiplier', 1.0, 51.0, step=5.0),
           'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
           'weight_decay': trial.suggest_float('weight_decay', 0.0, 0.1, step=0.01),
           'warmup_ratio': trial.suggest_float('warmup_ratio', 0.0, 0.3, step=0.05),
           'gradient_accumulation_steps': trial.suggest_categorical('gradient_accumulation_steps', [1, 2, 4]),
           'max_grad_norm': trial.suggest_float('max_grad_norm', 0.5, 2.0, step=0.5),
           'max_length': trial.suggest_categorical('max_length', [64, 128, 256]),
           'num_epochs': trial.suggest_int('num_epochs', 2, 10)
       }


       hidden_dims = []
       base_dim = 768
       for i in range(int(hp_dict['num_hidden_layers'])):
           dim = int(base_dim * float(hp_dict['hidden_dim_ratio']) * (0.5 ** i))
           hidden_dims.append(dim)


       train_dataset = QueryDataset(
           self.train_queries, self.train_scores, self.tokenizer, int(hp_dict['max_length'])
       )


       val_dataset = QueryDataset(
           self.val_queries, self.val_scores, self.tokenizer, int(hp_dict['max_length'])
       )


       train_dataloader = DataLoader(
           train_dataset,
           batch_size=int(hp_dict['batch_size']),
           shuffle=True,
           num_workers=0
       )


       val_dataloader = DataLoader(
           val_dataset,
           batch_size=int(hp_dict['batch_size']),
           shuffle=False,
           num_workers=0
       )


       model = quRewardModel(
           dropout=float(hp_dict['dropout']),
           hidden_dims=hidden_dims,
           activation=str(hp_dict['activation']),
           use_layer_norm=bool(hp_dict['use_layer_norm'])
       )


       trainer = quTrainer(
           model=model,
           device=self.device['device'],
           learning_rate=float(hp_dict['learning_rate']),
           head_lr_multiplier=float(hp_dict['head_lr_multiplier']),
           weight_decay=float(hp_dict['weight_decay']),
           warmup_ratio=float(hp_dict['warmup_ratio']),
           gradient_accumulation_steps=int(hp_dict['gradient_accumulation_steps']),
           max_grad_norm=float(hp_dict['max_grad_norm'])
       )


       val_loss = trainer.train(
           train_dataloader=train_dataloader,
           val_dataloader=val_dataloader,
           num_epochs=int(hp_dict['num_epochs']),
           save_path=None,
           verbose=False,
           early_stopping_patience=2
       )


       del model
       del trainer
       torch.cuda.empty_cache()


       return val_loss
```


<br>Last is your optimize function, which should choose your sampler, create the optuna study, and call the optimize function on it, passing in your objective function so it knows what to do. Afterwards you should write out the optimization history and return it to the caller so you can use this to extract the best parameters and validation loss as well as visualize how the process went to get an idea of where you might want to improve it (like if your convergence plot shows that you’re still getting significant improvements at the end, up your n\_trials and go again).

<br>

```
   def optimize(self) -> Dict[str, Any]:
       sampler = TPESampler(seed=42)
       study = optuna.create_study(
           direction='minimize',
           sampler=sampler,
           study_name=f'{self.model_id}_quRewardModel'
       )


       study.optimize(
           self.objective,
           n_trials=self.n_trials,
           timeout=self.timeout,
           show_progress_bar=True
       )


       best_params = study.best_params
       best_value = study.best_value


       logger.info(f"Best validation loss: {best_value:.4f}")
       logger.info(f"Best params: {best_params}")


       history = {
           'best_params': best_params,
           'best_value': best_value,
           'n_trials': len(study.trials),
           'optimization_history': [
               {
                   'trial': i,
                   'value': trial.value,
                   'params': trial.params
               }
               for i, trial in enumerate(study.trials)
               if trial.value is not None
           ]
       }


       return history
```


<br>Lastly, you need to actually use the class. Set your torch device, load your data and tokenizer, do your train/val/test split, create your optimizer and run it, read back the best parameters, train your final model on train+val and evaluate it on test, and then use the model\!

<br>

```
def main(model_id="answerdotai/ModernBERT-base"):
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   torch.set_float32_matmul_precision('high')
  
   tokenizer = AutoTokenizer.from_pretrained(model_id)


   train_df = pl.read_csv("./data/query-wellformedness/train.tsv", separator='\t', has_header=False, new_columns=["query", "score"])
   test_df = pl.read_csv("./data/query-wellformedness/test.tsv", separator='\t', has_header=False, new_columns=["query", "score"])


   queries_list = train_df.select("query").to_series().to_list()
   scores_list = train_df.select("score").to_series().to_list()


   train_len = floor(len(queries_list) * 0.8)


   train_queries = queries_list[:train_len]
   val_queries = queries_list[train_len:]


   train_scores = scores_list[:train_len]
   val_scores = scores_list[train_len:]


   test_queries = test_df.select("query").to_series().to_list()
   test_scores = test_df.select("score").to_series().to_list()


   print("Starting Bayesian Hyperparameter Optimization")


   if not os.path.exists(f'results/{model_id.replace("/", "_")}/optimization_history.json'):
       optimizer = BayesianOptimizer(
           train_queries=train_queries,
           train_scores=train_scores,
           val_queries=val_queries,
           val_scores=val_scores,
           tokenizer=tokenizer,
           device=device,
           n_trials=50,
           model_id=model_id
       )


       optimization_history = optimizer.optimize()


      
       os.makedirs(f'results/{model_id.replace("/", "_")}', exist_ok=True)
       with open(f'results/{model_id.replace("/", "_")}/optimization_history.json', 'w') as f:
           json.dump(optimization_history, f, indent=2)
   else:
       with open(f'results/{model_id.replace("/", "_")}/optimization_history.json', 'r') as f:
           optimization_history = json.load(f)


   final_model, final_val_loss = train_with_best_hyperparameters(
       train_queries=train_queries + val_queries,
       train_scores=train_scores + val_scores,
       val_queries=test_queries,
       val_scores=test_scores,
       best_params=optimization_history['best_params'],
       device=device,
       model_id=model_id
   )
  
   print(f'\nFinal validation loss: {final_val_loss:.4f}')
   print(f'\nOptimization complete! Best hyperparameters saved to results/{model_id.replace("/", "_")}/optimization_history.json')
  
   final_model.eval()
   test_query = "How does neural network training work?"
   with torch.no_grad():
       encoding = tokenizer(
           test_query,
           truncation=True,
           padding='max_length',
           max_length=optimization_history['best_params']['max_length'],
           return_tensors='pt'
       )
       input_ids = encoding['input_ids'].to(device)
       attention_mask = encoding['attention_mask'].to(device)
      
       score = final_model(input_ids, attention_mask)
       print(f'\nTest query: "{test_query}"')
       print(f'Predicted score: {score.item():.4f}')
```

<br>

### **Additional Materials**

If you want to explore these ideas further and in much finer detail than I have time to go over here, I recommend these playlists on YouTube that I learned from:

- [Neural Networks/Deep Learning from StatQuest with Josh Starmer](https://www.youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1)  
- [Convolutional Neural Networks from Stanford University](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)