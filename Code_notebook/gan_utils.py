import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, schedules
from keras.layers import Input
from keras.models import Model

from evaluate_utils import *

def create_discriminator(optimizer):
    discriminator = Sequential()
    
    #Add dropout to input
    discriminator.add(Dropout(0.1, input_shape=(300,)))

    #First hidden layer
    discriminator.add(Dense(2048))
    discriminator.add(LeakyReLU(0.2))

    #Second hidden layer
    discriminator.add(Dense(2048))
    discriminator.add(LeakyReLU(0.2))

    #Output layer
    discriminator.add(Dense(1, activation = 'sigmoid'))
   
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

    return discriminator

def create_generator(optimizer):
    generator = Sequential()
    
    #the generator or the mapping uses a 300x300 matrix to rotate the input (no bias)
    generator.add(Dense(300, use_bias = False, input_dim=300))
    
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)

    #We initialiaze the matrix to identiy
    generator.set_weights([np.eye(300)])

    return generator


def compile_gan(initial_learning_rate, decay_steps, decay_rate):

    word_dimension = 300

    lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps,
    decay_rate)

    optimizer = SGD(learning_rate=lr_schedule)


    discriminator = create_discriminator(optimizer)
    generator = create_generator(optimizer)

    discriminator.trainable = False

    gan_input = Input(shape=(word_dimension,))
    rotated_word = generator(gan_input)

    gan_output = discriminator(rotated_word)

    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)

    return discriminator, generator, gan


def train_gan(X_train, tgt_embeddings, X_val, Z_val, discriminator, generator, gan, epochs = 25, batch_size = 32, epoch_size = 5000, dis_steps = 3, beta = 0.001, sm = 0.1, mf = 15000):
  """
      X_train : src_embeddings
      tgt_embeddigs 
      X_val, Z_val: arrays of vectors that are the translation of each other (used for validation)
      discriminator, generator, gan : The different parts of our network obtained in compile_gan() function
      epochs = 25, 
      batch_size = 32
      epoch_size = 5000, 
      dis_steps = 3 : Nb of training loops for discriminator 
      beta = 0.001 : Orthogonalization parameter
      sm = 0.1 : Label smoothening  
      mf = 15000 : number of most frequent words to use
  """
  val_acc = []
  word_dimension = 300
  W_best = np.ones(300)
  best_val_acc = 0

  for epoch in range(epochs):
    print(f'Epoch: {epoch} ')
    for i in range(epoch_size):
      # prepare batch for generator
      # We take the most frequent words mf in order to have a more accurate embeddings
      word_batch = X_train[np.random.randint(0, mf, size=batch_size)]
      word_batch.reshape(batch_size,word_dimension)

      #We calculate the output of our mapping
      rotated_batch = generator.predict(word_batch)

      # dis_steps = 3 training the  discriminator
      for j in range(dis_steps) :
        # We choose batch_size sample from the most frequent words of the target embedding
        target_batch = tgt_embeddings[np.random.randint(0, mf, size=batch_size)]
        #We concatenate the 'fake data' with the 'true data'
        x = np.concatenate((rotated_batch,target_batch))
        disc_y = np.zeros(2*batch_size)
        #Label smoothening
        disc_y[:batch_size] = 1-sm
        disc_y[batch_size:] = sm
        d_loss = discriminator.train_on_batch(x, disc_y)

      # training the mapping
      y_gen = np.ones(batch_size)*sm
      g_loss = gan.train_on_batch(word_batch, y_gen)

      #We orthogonalize using the paper's method
      W = generator.get_weights()[0]
      W_orth =  (1 + beta)*W - beta*(W.dot(W.T)).dot(W)
      generator.set_weights([W_orth])


    print(f'      Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')
    #We compute accuracy on validation set
    acc = accuracy(X_val, Z_val, generator.get_weights()[0], tgt_embeddings, 1)
    val_acc.append(acc)
    print(f'      Accuracy of translation on validation set :    {acc} %\n')
    #We save the model's weights if it performs better on validation set
    if acc > best_val_acc :
      W_best = generator.get_weights()[0]
      best_val_acc = acc
      
  return val_acc, W_best