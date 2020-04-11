# udacity-self-driving-car
This is a machine learning project, in which a car is driven autonomously in a simulator using a nine-layered convolutional neural network. The simulator used in the video is Udacity's open source simulator. The video link is below:

<video src="video.mp4" width="320" height="200" controls preload><https://www.youtube.com/watch?v=9lgAEADN3Fk>

### Prerequisites

You'll require the following packages to run this on you own machine. 
## libraries
* [Udacity open source simulator](https://github.com/udacity/self-driving-car-sim)
* [Tensorflow-gpu](https://www.tensorflow.org/install/) - Deep learning library used.
* [TFlearn](https://tflearn.org/) - Higher level wrapper on tensorflow.
* you only need to install tensoflow by your self. Others can be isntalled using requirements.txt, follow the instructions in installing section 

### Installing
* clone this repository.
* [Tensorflow-gpu](https://www.tensorflow.org/install/) - Install tensorflow gpu from this link (you will need a nvidia graphics card)
* install opencv
* use the following command to install other dependencies.
```
pip install -r requirements.txt
```

if it fails you can install the following libraries using pip directly. (if you get stuck somewhere doing this, just google it)


## Checking the installation
To check the installation of the libraries you can import them in your python terminal. 
(mostly everything except tensorflow and opencv should work without any problems)

## How to train your own neural network
create IMG, data folder
save your images in IMG folder and save the driving_log.csv in data folder then run the following command.

```
python model_train.py
```

## Testing

Start the simulator in autonomous mode. Then run the following command.
```
python drive.py
```
## Authors

* **Navid Panchi** - [*IvLabs, Vnit nagpur*](http://www.ivlabs.in/) - [github](https://github.com/navidpanchi)

* **Archana Gahiwad** - [*IvLabs, Vnit nagpur*](http://www.ivlabs.in/) 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

