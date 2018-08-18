#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from numpy import random

RANDOM_SEED = 314
random.seed(RANDOM_SEED)

a = 10
b = -4
X_RANGE = (-10, 10)
ERROR_RANGE = (-20, 20)

a = tf.get_variable("a", shape=[1], dtype=tf.float32)
b = tf.get_variable("b", shape=[1], dtype=tf.float32)

x = tf.placeholder(name="x", dtype=tf.float32)
y = tf.placeholder(name="y", dtype=tf.float32)

square_error = tf.square(y - a*x - b)

rss = tf.reduce_sum(square_error)

optimizer = tf.train.GradientDescentOptimizer(1.0e-5)
minimize = optimizer.minimize(rss)