import keras as k
import tensorflow as tf
import numpy as np
import os
from keras import mixed_precision

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())  # OpenMP threads
os.environ["TF_NUM_INTRAOP_THREADS"] = str(os.cpu_count())  # per-op parallelism
os.environ["TF_NUM_INTEROP_THREADS"] = str(os.cpu_count())  # between op parallelism
tf.config.optimizer.set_jit(True)

callbacks = k.callbacks
utils = k.utils
models = k.models
layers = k.layers
optemizers = k.optimizers

X = np.load("Xinputs.npy", allow_pickle=True)
y = np.load("yInputs.npy", allow_pickle=True)
yEval = np.load("yEval.npy", allow_pickle=True)
intMoves = {}
intMoves = np.load("numTypeMoves.npy", allow_pickle=True)
intMoves = intMoves.item()

print(yEval[0])
XTrain, yTrain, yEvalTrain = X, y, yEval
print(XTrain.shape, yTrain.shape, yEval.shape, len(intMoves))
# XTrain,yTrain,yEvalTrain = XTrain[0:1000],yTrain[0:1000],yEvalTrain[0:1000]
print(XTrain.shape, yTrain.shape, yEval.shape, len(intMoves))
yTrain = utils.to_categorical(y, num_classes=len(intMoves))
inputLayer = k.Input((14, 8, 8))
x = layers.Permute((2, 3, 1))(inputLayer)

x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
x = layers.Conv2D(128, (3, 3), padding="same")(x)
x = layers.GlobalAveragePooling2D()(x)
e = layers.Dense(256, activation="relu")(x)
e = layers.Dense(64, activation="relu")(e)
e = layers.Dropout(0.2)(e)
EvalOutput = layers.Dense(1, activation="linear", name="eval")(e)

model = models.Model(inputs=inputLayer, outputs=[EvalOutput])
model.compile(
    optimizer=optemizers.AdamW(),
    loss={"eval": k.losses.Huber()},
    metrics={"eval": "mse"},
    loss_weights={"eval": 0.5},
)
model.summary()

model.fit(
    XTrain,
    yEvalTrain,
    epochs=20,
    validation_split=0.4,
    shuffle=True,
    batch_size=64,
    callbacks=[
        k.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6
        )
    ],
)
model.save("chessModelEval.keras")
