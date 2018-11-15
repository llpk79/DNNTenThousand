import tensorflow as tf
import numpy as np
import pandas as pd

from collections import Counter
from itertools import combinations_with_replacement as combos
from itertools import permutations as perms

import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from tensorflow.data import Dataset


tf.enable_eager_execution()
tfe = tf.contrib.eager


def is_three_pair(choice):
    choice = sorted(choice)
    return (len(choice) == 6 and choice[0] == choice[1] and
            choice[2] == choice[3] and choice[4] == choice[5])


def is_straight(choice):
    return sorted(choice) == list(range(1, 7))


def score_all():
    return [1.] * 6


def choose_dice(roll):
    """Choose dice according to scoring rules. Boop Beep."""
    counts = Counter(roll)
    if is_three_pair(roll) and (sum(scoring_rules[die - 1][count - 1] for die, count in counts.items()) < 1500):
        choice = score_all()
    elif is_straight(roll):
        choice = score_all()
    else:
        picks = set()
        for die, count in counts.items():
            if scoring_rules[die - 1][count - 1] > 0:
                picks.add(die)
        choice = [0.] * 6
        for i, x in enumerate(roll):
            if x in picks:
                choice[i] = 1.
    return np.array(choice)


scoring_rules = [[100, 200, 1000, 2000, 4000, 5000],
                 [0, 0, 200, 400, 800, 5000],
                 [0, 0, 300, 600, 1200, 5000],
                 [0, 0, 400, 800, 1600, 5000],
                 [50, 100, 500, 1000, 2000, 5000],
                 [0, 0, 600, 1200, 2400, 5000]
                 ]


def make_some_features(numbers, clip):
    features = set()
    combinations = (combo for combo in combos(numbers, 6))
    for i, comb in enumerate(combinations):
        if i % clip == 0:  # Keeping size reasonable
            for perm in perms(comb):
                features.add(perm)
    return features


features = make_some_features(list(range(1, 7)), 3)
special_features = set()
for _ in range(1000):
    half = [np.random.randint(1, 6) for _ in range(3)]
    half += half
    for perm in perms(half):
        special_features.add(perm)

for perm in perms([1, 2, 3, 4, 5, 6]):
    special_features.add(perm)

all_features = [np.array(feature) for feature in special_features]
all_features += [np.array(feature) for feature in features]
all_labels = [choose_dice(feature) for feature in special_features]
all_labels += [choose_dice(feature) for feature in features]


def create_dataset(features, labels):
    dice = pd.Series(features)
    labels = pd.Series(labels)
    dataset = pd.DataFrame({'dice': dice,
                            'labels': labels})
    return dataset


all_dice = create_dataset(all_features, all_labels)
all_dice = all_dice.reindex(np.random.permutation(all_dice.index))

train_dice = all_dice.head(10000)
val_dice = train_dice.tail(5000)
test_dice = all_dice.tail(1936)


def decode_feature(features):
    if features.shape == (1, 36):
        features = features[0].numpy().reshape((6, 6))
    output = []
    for feature in features:
        output.append(tf.argmax(feature).numpy() + 1)
    return output


def decode_label(labels):
    guessed_label = []
    if len(labels.shape) > 2:
        labels = tf.squeeze(labels, [0])
    for label in labels:
        if label.numpy()[0] > 0.:
            guessed_label.append(1.)
        else:
            guessed_label.append(0.)
    return guessed_label


def pre_process_features(dice: pd.DataFrame) -> list:
    rolls = []
    for roll in dice['dice']:
        roll = np.array(roll)
        roll -= 1
        roll = tf.one_hot(roll, depth=6, axis=-1)
        rolls.append(roll)
    return rolls


def pre_process_labels(dice: pd.DataFrame) -> list:
    labels = [tf.reshape(tf.convert_to_tensor([label]), (6, 1)) for label in dice['labels']]
    return labels


def test_f_l_pairs(features, labels):
    for feature, label in zip(features, labels):
        if tuple(choose_dice(decode_feature(feature))) != tuple(label.numpy()):
            print(feature, label)
            return False
    return True


test_features = pre_process_features(test_dice)
test_labels = pre_process_labels(test_dice)
# print(test_f_l_pairs(test_features, test_labels))

train_features = pre_process_features(train_dice)
train_labels = pre_process_labels(train_dice)
# print(test_f_l_pairs(train_features, train_labels))

val_features = pre_process_features(val_dice)
val_labels = pre_process_labels(val_dice)
# print(test_f_l_pairs(val_features, val_labels))

predict_dice = train_dice.tail(1)
predict_features = pre_process_features(predict_dice)
predict_labels = pre_process_labels(predict_dice)
# print(test_f_l_pairs(predict_features, predict_labels))

num_epochs = 2001
learning_rate = 0.0001
regularization_rate = 0.6
batch_size = 64


def make_dataset(features, labels):
    dataset = Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(batch_size=batch_size).shuffle(buffer_size=10000).repeat()
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels


test_features, test_labels = make_dataset(test_features, test_labels)

train_features, train_labels = make_dataset(train_features, train_labels)

val_features, val_labels = make_dataset(val_features, val_labels)

predict_features, predict_labels = make_dataset(predict_features, predict_labels)


model = tf.keras.Sequential([
    layers.Dense(6, activation=tf.nn.relu, input_shape=(6, 6),
                 kernel_regularizer=tf.keras.regularizers.l2(regularization_rate)),
    layers.Dense(64, activation=tf.nn.relu,
                 kernel_regularizer=tf.keras.regularizers.l2(regularization_rate)),
    layers.Dense(128, activation=tf.nn.relu,
                 kernel_regularizer=tf.keras.regularizers.l2(regularization_rate)),
    # layers.Dense(256, activation=tf.nn.relu,
    #              kernel_regularizer=tf.keras.regularizers.l2(regularization_rate)),
    layers.Dense(32, activation=tf.nn.relu,
                 kernel_regularizer=tf.keras.regularizers.l2(regularization_rate)),
    layers.Dense(1)])

optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
global_step = tf.train.get_or_create_global_step()


#testing
predictions = model(predict_features)
tf.nn.sigmoid(predictions)
predicted_label = []
for prediction in predictions[0]:
    if prediction[0].numpy() > 0.:
        predicted_label.append(1)
    else:
        predicted_label.append(0)
print(f'Feature: {decode_feature(predict_features.numpy()[0])}')
print(f'True Label:      {predict_labels[0]}')
print(f'Predicted Label: {predicted_label}')


def loss(model, features, labels):
    logits = model(features)
    if logits.shape == (1, 6, 1):
        logits = tf.squeeze(logits, [0])
    standard_loss = tf.losses.sigmoid_cross_entropy(logits=logits, multi_class_labels=labels)
    return standard_loss


def grad(model, features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss(model, features, labels)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


loss_value, grads = grad(model, train_features, train_labels)
print(f'Step {global_step.numpy()} Initial loss: {loss_value.numpy()}')

optimizer.apply_gradients(zip(grads, model.variables), global_step)
print(f'Step {global_step.numpy()} Loss: {loss(model, train_features, train_labels)}')


train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []

val_features, val_labels = iter(val_features), iter(val_labels)
val_feature, val_label = next(val_features), next(val_labels)
for epoch in range(num_epochs):
    epoch_loss_ave = tfe.metrics.Mean('loss')
    epoch_val_loss_average = tfe.metrics.Mean('loss')
    epoch_accuracy = tfe.metrics.Accuracy('acc')
    epoch_val_accuracy = tfe.metrics.Accuracy('acc')

    for feature, label in zip(train_features, train_labels):
        feature = tf.convert_to_tensor(feature.numpy().reshape(1, 6, 6))

        loss_value, grads = grad(model, feature, label)
        optimizer.apply_gradients(zip(grads, model.variables), global_step)
        epoch_loss_ave(loss_value)

        guessed_label = decode_label(model(feature))
        epoch_accuracy(guessed_label, decode_label(label))

        val_loss_value = loss(model, val_feature, val_label)
        epoch_val_loss_average(val_loss_value)

        val_guess_label = decode_label(model(val_feature))
        epoch_val_accuracy(val_guess_label, decode_label(val_label))

    train_loss.append(epoch_loss_ave.result())
    train_accuracy.append(epoch_accuracy.result())

    val_loss.append(epoch_val_loss_average.result())
    val_accuracy.append((epoch_val_accuracy.result()))

    if epoch % 20 == 0:
        print(f'Epoch {epoch} Loss: {epoch_loss_ave.result()} Accuracy: {epoch_accuracy.result()}')
        print(f'Validation loss: {epoch_val_loss_average.result()} Accuracy: {epoch_val_accuracy.result()}')


plt.plot(train_loss, label='Train Loss')
plt.title('Loss/Accuracy by Epoch')
plt.plot(val_loss, label='Val Loss')
plt.plot(train_accuracy, label='Train Acc')
plt.plot(val_accuracy, label='Val Acc')
plt.legend()
plt.show()


test_results = []
test_accuracy = tfe.metrics.Accuracy('acc')

for feature, label in zip(test_features, test_labels):

    guessed_label = decode_label(model(feature))
    test_accuracy(guessed_label, decode_label(label))
print(f'Test accuracy: {test_accuracy.result()}')

for _ in range(25):
    roll = np.array([np.random.randint(0, 5) for _ in range(6)])
    turn = tf.one_hot(roll, depth=6, dtype=np.int32)
    roll += 1
    answer = choose_dice(roll)
    print(f'Roll: {roll}')
    print(f'Dice expected to be kept:  {answer}')
    turn = tf.convert_to_tensor(turn.numpy().reshape((1, 6, 6)), dtype=tf.float32)
    predictions = model.predict(turn)
    tf.nn.softmax(predictions)
    predicted_label = []
    for prediction in predictions[0]:
        if prediction[0] > 0.:
            predicted_label.append(1.)
        else:
            predicted_label.append(0.)
    print(f'Dice predicted to be kept: {predicted_label}')
