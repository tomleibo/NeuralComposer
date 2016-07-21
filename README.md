# NeuralComposer

## Data Science Course - Final Project 
- Manor Lahagani 201049376
- Tom Leibovich 200456267

## Sections 1 & 2 - Pattern Data
 - As we first started the project we had to choose a type of pattern data. As we both love music and it's a part of our everyday life we were curious about what can deep learning do in that field. 
 - The data collection part was therefore quite easy, as we all have songs on our PCs and phones. There are many music collections in the internet but we couldn't use a big number of songs (exaplained below), so we just picked a few.
- Challenges we faced while working with the data:
  1. mp3 is a complex format which can not be read and converted to features easily, therefore we first converted the data into wav format.
  1. Wav format consists of chunks of data which is the frequency of the music as a time series. Another challenge was converting it into numpy arrays of floats.
  1. The input created from several songs was too big (several gigabytes) and therefore the computation never finished. We has to pick just a few songs.
  1. Using many hidden layers led to "out of memory" exceptions, therefore we chose to use 3 layers only.
  1. Running Theano and Keras required using the GPU. Because our laptops don't have high quality GPUs that can integrate with CUDA, we used AWS' GPU machines. That required learning EC2, S3 and how to work with them.
 - Contribution - The project doesn't much contribute to society but is mostly for entertainment and to discover the capabilities of deep learning in the field of music generation.

## Section 3 - pre-processing
Pre-processing stages we applied on the raw data:
 1. As mentioned above, mp3 is a complex and compressed format which can not be read easily therefore we had to convert to wav and then to convert from wav to array of floats representing sound frequencies.
 2. Converting from mp3 to wav was not in the scope of the project and we used an external library called lame. installing is as easy as: sudo apt-get install lame
 3. After converting to wav, reading it was a lot easier. We used the library scipy.io.wavfile which returns a numpy array with meta data and byte arrays

## Sections 4, 5 & 6  - The Code and building instructions
The code was generated in a python notebook which ran on an AWS machine and can be found on https://github.com/tomleibo/NeuralComposer

### Building and Running
 - Create an AWS EC2 instance with the following specification:
   1. Region - North California / us-west-1
   2. Community AMI - f7bd7bb3
   3. Instance type - g2.8xlarge
 - SSH into the machine and run the following commands:
      1. git clone https://github.com/tomleibo/NeuralComposer
      2. cd NeuralComposer
      3. sudo apt-get update
      4. sudo apt-get install awscli
      5. sudo apt-get install lame
      6. cd datasets/
      7. cd songs/
      8. rm *
      9. copy any songs you want to train on. we provided a few songs on our bucket:
           - aws s3 cp s3://deep-learn-files/songs/* ./ --region us-west-1 
      10. cd ../..
      11. python notebook.py
      12. wait a few minutes
      13. voila! your new song is generated in the directory and is called: out.wav

## Section 7 - 
 
## Section 8 - Results & Conclusions
 - The results of the task - the generated songs, can be found on the repository
 - Conclusions -
    1. Deep learning requires a lot of computation power. Our project required a big amount of GBs of RAM, GPU, time and money (running on AWS). These 3 limitations led to compromising on the quality of the neural network. We compromised on:
        - The number of hidden layers. The optimal configuration would include at least a few tens of hidden layers.
         - The size of each hidden layer (dimensions) - An optimal configuration would include layers which are approximately the size of the input layer. We succeeded to run with 100.
         - The number of songs trained on - Using 15 songs created very large inputs (3 GBs). An optimal configuration would include hundreds or thousands of songs. We ran the algorithm on one song. 
         - The time we ran the algorithm - 
    2. Deep Learning can run in many configurations. Thorough research is needed in order to find the algorithm which creates the best results. Some of the configurations we looked into: 
       - Different Algorithms. Keras enables three recurrent layer classes. 
          1. SimpleRNN.
          2. GRU.
          3. LSTM.
       - Layers - The number of layers can be configured, each with its own dimensions and type. 
       - Dimensions - The first and the last layers have to be in the shape of the input. But the hidden layers can be configured to whichever number one chooses.
       - Recurrent input - The output can be inserted as input in many ways. We chose to insert it as half of the input alongside the original input.
       - Loss function - different loss function can be set. We used the default setting.
       - Activation functions - different activation functions can be set. We used the default setting.
      
  
