# DNNTenThousand
A neural network for predicting best dice to keep in a given throw in Ten Thousand dice game.

I have an open question about this code at [DataScience on Stack Exchange](https://datascience.stackexchange.com/questions/41723/engineering-features-for-dice-game-predictions)

Tensorflow model for predicting which dice to keep for a given roll. Thus far it works well for 
predicting 1's and 5's, but not three of a kind, three pairs or straights. I may need to use a 
different optimizer, or add more examples of special scoring situations to the training set.
I'll ask at Stack Exchange Data Science. 
