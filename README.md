# Seedtag Codetest

## Problem description
I focused on finding a solution for the two challenges presented in the
 document `problem.pdf`.
- Challenge 1: given a model for image recognition, hack it to miss a
 labrador retriever image changing the pixels in a way not detectable for
  the human eye.
- Challenge 2: training a flower recognition model strong enough against the hack from
 Challenge 1
 
 ## Tools
 - Python (see requirements.txt for dependencies)
 - Makefile
 - Colab from Google
 
 ## Usage
 There is a makefile for Challenge 1:  
  `make install_<unix/windows>` will set up venv  
  `make run` will execute
   model.py to test the labrador retriever hack  
   I can guarantee that this
    will work on your side. 
  
 There is a notebook `flower_power.ipynb` made in Colab for Challenge 2: please open, take into
  account paths, and review. I cannot guarantee that this will work on your
   side.
 
 ## Solution
 ### Challenge 1: the labrador retriever hack
 #### 1. Thinking
 When I first read the problem I opened a Colab notebook and tried the obvious
  just for starters: let's just change randomly all pixels in the image to
   +/-1 and see what happens, but precisely, it was too random.  
   Then I started reading about adversarial images and found three
    interesting approaches:  
- The Adversarial Patch: which consists in adding an ugly patch to the
     input image to make the network fail.
- The One Pixel Attack: which claims that modifying only one precise pixel
 of the input image will make the network fail.
- The Fast Gradient Signed Method (FGSM): which adds noise to the input
 image by using
 gradients of the trained network, to make it fail.
 
The Adversarial Patch was discarded right away since it is too obvious and
 visible for the human eye, and we want the change to be undetected.  
 I tried
  changing one pixel only at random areas of the image
 (center, up/down
-left/right quarter centers) and it did something (few decimal points lower
 than the original precision). So I find this approach very interesting
  because it does not depend on the pretrained model itself. Finding the
   correct pixel is a rather complex optimization problem, so I discarded
    this approach too.
     
 #### 2. Developing
 FGSM consists on taking the pretrained model, apply it to the input image, calculate gradients of the loss functions and use their signs to make
  some blurry noise and add it on top of the original image. This way, we do
   not modify the pretrained model weigths nor the internal layers
    configuration, but rather maximize the loss by learning the
     contribution of each input pixel.
     
So, I coded a class `Fool` that can help to deceive the model given an input
 image related to a specific class of the ImageNet (https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), and the predicted result obtained previously
  with the pretrained model.
In this case, we know that the input image is a labrador retriever, whose
 imagenet index is 208. To test it with other images, it is necessary to
  change manually `thing_index` in `model.py` to the corresponding image class.
 
I use this class to calculate the noise square using gradients of the loss, and
 patch it to the input image weighted by an epsilon (0.01). The larger is
  the epsilon the more detectable the change is to the human eye. Then, the
   modified image is re-predicted and served as the output. `model.py` plots
    this latter result instead of the initially obtained prediction.
 
 It is commented in the code, but it is possible to plot the noise and the
  fake image obtained as adversarial to see how it changes in respect to the
   original image. This compartmentalization of the code will be useful in
    the Challenge 2.
    
I wanted to write something simple that could easily be added to the
 provided code, so I chose to build a new class usable in the main.
 
 #### 3. Results
When executing `model.py` with `EPSILON=0.01`, the labrador retriever is
 classified as a Saluki
 with 16.60% confidence (saved in `/assets` folder), which is another type of
  hound dog
  (imagenet index 176). If `EPSILON=0.15`, the model thinks the image is a Weimaraner, and if
  `EPSILON=0.2` the model even thinks the image is an African chameleon
   (with low levels of confidence).


 
 ### Challenge 2: detect flowers
 #### 1. Thinking
 This one was relatively "easy" to decide. I just went for a transfer
  learning approach and supported my coding with available repositories.
  I downloaded 
 Prior to anything, given the bulk structure of the flowers dataset I had to
  download the flowers into my local disk and then
  run some bash code to split it into train and test sets using path
   directories to infer the classes
   (you can find the bash script in `/examples/flowers.sh`).  
   In this problem
    the key is data augmentation: besides the normal flips, rotations, etc. we should use new generated images containing the noise
 built in Challenge 1. Meaning: we can use `_make_noise()` from class `Fool`
  on all those flower images in train folder, in order to include additional
   adversarial images in
   the training process. This way our model would learn also to ignore the
    FGSM hack. Sorry I did not implement this idea because time.
The generators for train and test also take care of homogeneous distribution
 through flower classes.

   
 #### 2. Developing
After the transfer learning, and adding some main layers at the end, the
 model is ready to recognize 5 kinds of flowers: daisies, dandelions, roses, tulips and sunflowers (which are my favourite).
Further details of the analysis and cells can be found inside the markdowns
 of the notebook.
 Using GPUs in Colab I was able to train the model three times to see if I
 could improve by increasing epochs and substeps.

 #### 3. Results
The best version of the model v3 took 50 mins to train (accuracy is 91.25
% and was reached at
 epoch 15th) and is saved in Colab folder so
 you can load it and test it at the end of the notebook. I tried with some
  random
  pictures from Google and some photos
  I took in the _Jardín Botánico de Madrid_ at the beginning of Spring'21. It failed to predict three of them and succeeded in 10 of them.  
  I have the feeling (by looking at the history of training epochs) that this
   model v3 could be undertrained or overfitted
   to the train-test samples. Model versions v1 and v2 had lower accuracy
    but responded better to the out-of-sample test images. All three
     versions of the model can be found in `/assets` folder.


 ## Conclusions and improvements
- It was a lot of fun researching, reading, understanding and implementing
 the solutions. I felt motivated when I started unraveling the problem :)
 - I would like to learn how to optimize One Pixel Attack, it
 is something I did not fully understand in terms of implementation.
 - I should try to make a flower model from scratch and build it step by
  step to experiment the full picture of a neural network configuration.
 - I need to write a script that goes over every image in the train and test
  folders and generates adversarial images using `_make_noise()` from class
   `Fool
  ` and places them in the same class directory than the original ones. After that, I have to execute the notebook all over again and test the
   model against FGSM fake images.
- The notebook approach is a risk in terms of environments and
 reproducibility of results. I should find a more portable solution.
 
 ## References
 ### Challenge 1
- T.B. Brown, D. Mané, A. Roy, M. Abadí and J. Guiler, "Adversarial Patch
" in arXiv:1712.09665v2, May 2018, url: https://arxiv.org/abs/1712.09665
- J. Su, D. V. Vargas and K. Sakurai, "One Pixel Attack for Fooling Deep Neural Networks" in IEEE Transactions on Evolutionary Computation, vol. 23, no. 5, pp. 828-841, Oct. 2019, doi: 10.1109/TEVC.2019.2890858.
- I.J. Goodfellow, J. Shlens, C. Szegedy, "Explaining and Harnessing
 Adversarial Examples" in arXiv:1412.6572v3, Mar. 2015, url: https://arxiv.org/abs/1412.6572v3
- https://www.tensorflow.org/tutorials/generative/adversarial_fgsm
- https://towardsdatascience.com/how-to-systematically-fool-an-image-recognition-neural-network-7b2ac157375d
- https://buzzrobot.com/4-ways-to-easily-fool-your-deep-neural-net-dca49463bd0 
 ### Challenge 2
 - https://www.tensorflow.org/hub/tutorials/image_feature_vector
 - https://www.tensorflow.org/tutorials/images/data_augmentation
 - https://github.com/oleksandr-g-rock/flower_classification/blob/main/flowers_5_classes.ipynb
