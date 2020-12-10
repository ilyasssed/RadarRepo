# Report Project : Predicting Trafic volume using neural networks

Authors: Ilyass Seddoug

## Data preprocessing

After importing data and taking a look at it, we can see that we have columns for years, months and day. So, it would be better if we sorted the lines by years, months then days. After that, we transform our qualitative columns into integer classes so we can use them for our neural networks. We separate our labels and features into 2 variables, X for the features and y for the labels which is the variable to predict, the traffic volume. 
As a first step, we take only a small ammount of data to speed up te computations and test our models  on this small amount. After testing our models, and if we find that we may need more data, we can add data and train our models again. After that we scale the data to bring the vaues between 0 and 1, so that the scale of the data doesn't affect the convergence of our model, since neural networks are sensitive to the scales, and larger values may take better importance if not scaled. For that we use the minmaxscaler from sklearn.
First of all, we define a function SelectDataset to create our dataset based on our feature and labels. We transform our data to Pytorch tensors, we create dataloaders for training and for testing and then we are ready to begin working on our models.

## Deep Feed-Forward Neural Network
We build a model with multiple hidden layers that we train and test for 200 epochs. We choose touse multiple layers to avoid the problem of overfitting and have a better  generalization. 

```python
class NN(nn.Module):
    
    def __init__(self):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(11,5)
        self.fc2 = nn.Linear(5,5)
        self.fc3 = nn.Linear(5,1)
        
    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
```

## Convolutional neural network

## Reccurent Neural Network

- The syntax is very simple: it's just raw texts with special formatting for titles, images, equations, etc.
- Get a look [here](https://www.markdownguide.org/cheat-sheet/) for a quick guide about the syntax
- Including an image like this: <img src="CNN Losses.png" width="50%"/> (*attribution: XKCD.com*)
- Including a table is a bit annoying, but most good editors have plugins to easily edit markdown tables:

| Left-aligned | Center-aligned | Right-aligned |
| :---         |     :---:      |          ---: |
| git status   | git status     | git status    |
| git diff     | git diff       | git diff      |

- For a quotation with reference: "...the **go to** statement should be abolished..." [[1]](#1).

- If you want a PDF version of your report, you may convert it with pandoc:

```
   pandoc -s report.md -o report.pdf
```

or, if you use the img tag, you have to first convert into pure html:

```
pandoc -t html report.md | pandoc -f html -o report.pdf
```

## References
<a id="1">[1]</a> 
Dijkstra, E. W. (1968). 
Go to statement considered harmful. 
Communications of the ACM, 11(3), 147-148.

