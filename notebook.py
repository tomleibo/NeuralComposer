from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
from keras.models import Sequential
from keras.layers.core import TimeDistributedDense
from keras.layers.recurrent import SimpleRNN
import scipy.io.wavfile as wav
from pipes import quote
from scipy.spatial import distance

# # NeuralComposer
#
# ## Data Science Course - Final Project
# - Manor Lahagani 201049376
# - Tom Leibovich 200456267
#
# ## Sections 1 & 2 - Pattern Data
#  - As we first started the project we had to choose a type of pattern data. As we both love music and it's a part of our everyday life we were curious about what can deep learning do in that field.
#  - The data collection part was therefore quite easy, as we all have songs on our PCs and phones. There are many music collections in the internet but we couldn't use a big number of songs (exaplained below), so we just picked a few.
#  - Challenges we faced while working with the data:
#   1. mp3 is a complex format which can not be read and converted to features easily, therefore we first converted the data into wav format.
#   1. Wav format consists of chunks of data which is the frequency of the music as a time series. Another challenge was converting it into numpy arrays of floats.
#   1. The input created from several songs was too big (several gigabytes) and therefore the computation never finished. We has to pick just a few songs.
#   1. Using many hidden layers led to "out of memory" exceptions, therefore we chose to use 3 layers only.
#   1. Running Theano and Keras required using the GPU. Because our laptops don't have high quality GPUs that can integrate with CUDA, we used AWS' GPU machines. That required learning EC2, S3 and how to work with them.
#  - Contribution - The project doesn't much contribute to society but is mostly for entertainment and to discover the capabilities of deep learning in the field of music generation.
#
# ## Section 3 - pre-processing
# Pre-processing stages we applied on the raw data:
#  1. As mentioned above, mp3 is a complex and compressed format which can not be read easily therefore we had to convert to wav and then to convert from wav to array of floats representing sound frequencies.
#  2. Converting from mp3 to wav was not in the scope of the project and we used an external library called lame. installing is as easy as: sudo apt-get install lame
#  3. After converting to wav, reading it was a lot easier. We used the library scipy.io.wavfile which returns a numpy array with meta data and byte arrays
#
# ## Sections 4, 5 & 6  - The Code and building instructions
# The code was generated in a python notebook which ran on an AWS machine and can be found on https://github.com/tomleibo/NeuralComposer
#
# ### Building and Running
#  - Create an AWS EC2 instance with the following specification:
#    1. Region - North California / us-west-1
#    2. Community AMI - f7bd7bb3
#    3. Instance type - g2.8xlarge
#  - SSH into the machine and run the following commands:
#       1. git clone https://github.com/tomleibo/NeuralComposer
#       2. cd NeuralComposer
#       3. sudo apt-get update
#       4. sudo apt-get install awscli
#       5. sudo apt-get install lame
#       6. cd datasets/
#       7. cd songs/
#       8. rm *
#       9. copy any songs you want to train on. we provided a few songs on our bucket:
#            - aws s3 cp s3://deep-learn-files/songs/* ./ --region us-west-1
#       10. cd ../..
#       11. python notebook.py
#       12. wait a few minutes
#       13. voila! your new song is generated in the directory and is called: out.wav
#
# ## Section 8 - Results & Conclusions
#  - The results of the task - the generated songs, can be found on the repository
#  - Conclusions -
#     1. Deep learning requires a lot of computation power. Our project required a big amount of GBs of RAM, GPU, time and money (running on AWS). These 3 limitations led to compromising on the quality of the neural network. We compromised on:
#         - The number of hidden layers. The optimal configuration would include at least a few tens of hidden layers.
#          - The size of each hidden layer (dimensions) - An optimal configuration would include layers which are approximately the size of the input layer. We succeeded to run with 100.
#          - The number of songs trained on - Using 15 songs created very large inputs (3 GBs). An optimal configuration would include hundreds or thousands of songs. We ran the algorithm on one song.
#          - The time we ran the algorithm -
#     2. Deep Learning can run in many configurations. Thorough research is needed in order to find the algorithm which creates the best results. Some of the configurations we looked into:
#        - Different Algorithms. Keras enables three recurrent layer classes.
#           1. SimpleRNN.
#           2. GRU.
#           3. LSTM.
#        - Layers - The number of layers can be configured, each with its own dimensions and type.
#        - Dimensions - The first and the last layers have to be in the shape of the input. But the hidden layers can be configured to whichever number one chooses.
#        - Recurrent input - The output can be inserted as input in many ways. We chose to insert it as half of the input alongside the original input.
#        - Loss function - different loss function can be set. We used the default setting.
#        - Activation functions - different activation functions can be set. We used the default setting.

input_directory = './datasets/songs'
output_dir = './datasets/input'
print ('input directory',input_directory)
print('output_dir', output_dir)

# CONVERSION

freq = 44100
clip_len = 10
block_size = freq / 4 
max_seq_len = int(round((freq * clip_len) / block_size))
print ('Starting conversion from mp3 to wav and from wav to np.arrays')
for file in os.listdir(input_directory):
	fullfilename = input_directory+file
	cmd = 'lame -a -m m ' + fullfilename + ' ' + output_dir + '/tmp/' + file
	os.system(cmd)
	cmd = 'lame --decode fullfilename ' + output_dir + '/wave/' + file + ' --resample ' + str(float(freq) / 1000.0)
	os.system(cmd)
x_train,y_train,x_var,x_mean  = convert_wav_files_to_nptensor(output_dir + '/wave/', block_size, max_seq_len, output_dir)

#TRAIN:
# This part should be more complex but isnt because of computation limitations
# 1. The output should be inserted as part of the input for the next iteration.
# 2. more layers should be add to the sequential model.
# 3. layer dimension can be increased.
# 4. number of iterations can be increased
iteration = 0
model_basename = './TrainedWeights'
model_filename = model_basename + str(iteration)
hidden_layers = 1024
model =Sequential()
model.add(SimpleRNN(input_dim=x_train.shape[2], output_dim=hidden_layers))
model.add(SimpleRNN(input_dim=hidden_layers, output_dim=hidden_layers, return_sequences=True))
model.add(SimpleRNN(input_dim=hidden_layers, output_dim=x_train.shape[2]))
model.compile()

num_iters = 500
epochs_per_iter = 25	
batch_size = 5			
print ('Starting train:')
while iteration < num_iters:
	print('Iteration: ' + str(iteration))
	model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epochs_per_iter, verbose=1, validation_split=0.0)
	iteration += epochs_per_iter
print ('Finished train')
model.save_weights(model_basename + str(iteration))

#GENERATION

model_basename = './TrainedWeights'
model_filename = model_basename + str(iteration)
output_dir = './out.wav'

model =Sequential()
model.add(SimpleRNN(input_dim=x_train.shape[2], output_dim=hidden_layers))
model.add(SimpleRNN(input_dim=hidden_layers, output_dim=hidden_layers, return_sequences=True))
model.add(SimpleRNN(input_dim=hidden_layers, output_dim=x_train.shape[2]))
model.compile()

model.load_weights(model_filename)

print ('Generating:')

seed_len = 1
num_examples = x_train.shape[0]
example_len = x_train.shape[1]
randIdx = np.random.randint(num_examples, size=1)[0]
randSeed = np.concatenate(tuple([x_train[randIdx + i] for i in xrange(seed_len)]), axis=0)
seedSeq = np.reshape(randSeed, (1, randSeed.shape[0], randSeed.shape[1]))

max_seq_len = 10; 
output = []
for it in xrange(max_seq_len):
	seedSeqNew = model._predict(seedSeq)
	if it == 0:
		for i in xrange(seedSeqNew.shape[1]):
			output.append(seedSeqNew[0][i].copy())
	else:
		output.append(seedSeqNew[0][seedSeqNew.shape[1]-1].copy())
	newSeq = seedSeqNew[0][seedSeqNew.shape[1]-1]
	newSeq = np.reshape(newSeq, (1, 1, newSeq.shape[0]))
	seedSeq = np.concatenate((seedSeq, newSeq), axis=1)
for i in xrange(len(output)):
	output[i] *= X_var
	output[i] += X_mean
save_generated_example(output_dir, output, sample_frequency=freq)
print('output file saved as', output_dir)


def read_wav_as_np(filename):
	data = wav.read(filename)
	np_arr = data[1].astype('float32') / 32767.0 #Normalize 16-bit input to [-1, 1] range
	#np_arr = np.array(np_arr)
	return np_arr, data[0]

def write_np_as_wav(X, sample_rate, filename):
	Xnew = X * 32767.0
	Xnew = Xnew.astype('int16')
	wav.write(filename, sample_rate, Xnew)
	return

def convert_np_audio_to_sample_blocks(song_np, block_size):
	block_lists = []
	total_samples = song_np.shape[0]
	num_samples_so_far = 0
	while(num_samples_so_far < total_samples):
		block = song_np[num_samples_so_far:num_samples_so_far+block_size]
		if(block.shape[0] < block_size):
			padding = np.zeros((block_size - block.shape[0],))
			block = np.concatenate((block, padding))
		block_lists.append(block)
		num_samples_so_far += block_size
	return block_lists

def convert_sample_blocks_to_np_audio(blocks):
	song_np = np.concatenate(blocks)
	return song_np

def time_blocks_to_fft_blocks(blocks_time_domain):
	fft_blocks = []
	for block in blocks_time_domain:
		fft_block = np.fft.fft(block)
		new_block = np.concatenate((np.real(fft_block), np.imag(fft_block)))
		fft_blocks.append(new_block)
	return fft_blocks

def fft_blocks_to_time_blocks(blocks_ft_domain):
	time_blocks = []
	for block in blocks_ft_domain:
		num_elems = block.shape[0] / 2
		real_chunk = block[0:num_elems]
		imag_chunk = block[num_elems:]
		new_block = real_chunk + 1.0j * imag_chunk
		time_block = np.fft.ifft(new_block)
		time_blocks.append(time_block)
	return time_blocks


def convert_wav_files_to_nptensor(wav_dir, block_size, max_seq_len, output_filename, ):
	max_files=20
	files = []
	for file in os.listdir(wav_dir):
		if file.endswith('.wav'):
			files.append(wav_dir + file)
	chunks_X = []
	chunks_Y = []
	num_files = len(files)
	if(num_files > max_files):
		num_files = max_files
	for file_idx in xrange(num_files):
		file = files[file_idx]
		print ('Processing: ', (file_idx+1),'/',num_files)
		print ('Filename: ', file)
		X, Y = load_training_example(file, block_size)
		cur_seq = 0
		total_seq = len(X)
		print (total_seq)
		print (max_seq_len)
		while cur_seq + max_seq_len < total_seq:
			chunks_X.append(X[cur_seq:cur_seq+max_seq_len])
			chunks_Y.append(Y[cur_seq:cur_seq+max_seq_len])
			cur_seq += max_seq_len
	num_examples = len(chunks_X)
	num_dims_out = block_size * 2
	out_shape = (num_examples, max_seq_len, num_dims_out)
	x_data = np.zeros(out_shape)
	y_data = np.zeros(out_shape)
	print ('')
	for n in xrange(num_examples):
		for i in xrange(max_seq_len):
			x_data[n][i] = chunks_X[n][i]
			y_data[n][i] = chunks_Y[n][i]
	mean_x = np.mean(np.mean(x_data, axis=0), axis=0) #Mean across num examples and num timesteps
	std_x = np.sqrt(np.mean(np.mean(np.abs(x_data-mean_x)**2, axis=0), axis=0)) # STD across num examples and num timesteps
	std_x = np.maximum(1.0e-8, std_x) #Clamp variance if too tiny
	x_data[:][:] -= mean_x #Mean 0
	x_data[:][:] /= std_x #Variance 1
	y_data[:][:] -= mean_x #Mean 0
	y_data[:][:] /= std_x #Variance 1
	return x_data,y_data,std_x,mean_x

def convert_nptensor_to_wav_files(tensor, indices, filename):
	num_seqs = tensor.shape[1]
	for i in indices:
		chunks = []
		for x in xrange(num_seqs):
			chunks.append(tensor[i][x])
		save_generated_example(filename+str(i)+'.wav', chunks)

def load_training_example(filename, block_size=2048):
	data, bitrate = read_wav_as_np(filename)
	x_t = convert_np_audio_to_sample_blocks(data, block_size)
	y_t = x_t[1:]
	y_t.append(np.zeros(block_size)) #Add special end block composed of all zeros
	X = time_blocks_to_fft_blocks(x_t)
	Y = time_blocks_to_fft_blocks(y_t)
	return X, Y

def save_generated_example(filename, generated_sequence, freq=44100):
	time_blocks = fft_blocks_to_time_blocks(generated_sequence)
	song = convert_sample_blocks_to_np_audio(time_blocks)
	write_np_as_wav(song, freq, filename)
	return



