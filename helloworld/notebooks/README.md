# TransmogrifAI on Jupyter

In this section we will look at how TransmogrifAI can be run within Scala notebooks on
Jupyter.

We are going to leverage [BeakerX](http://beakerx.com/) Scala kernel for Jupyter

## Setup BeakerX on Linux / Ubuntu

Prequisities :

* Apache Maven
* Python 3
* JDK 8

Installation using pip

```$xslt
sudo pip install beakerx
sudo beakerx install
```

Installation using conda

```$xslt
conda create -y -n beakerx 'python>=3'
source activate beakerx
conda config --env --add pinned_packages 'openjdk>8.0.121'
conda install -y -c conda-forge ipywidgets beakerx
```

Reference : [BeakerX Documentation](http://beakerx.com/documentation)

## Setup BeakerX on Mac with Docker

BeakerX provides a [docker container image](https://hub.docker.com/r/beakerx/beakerx/) on docker hub. 

Assuming your Transmogrify source code is downloaded at `/Users/rdua/work/github/rajdeepd/TransmogrifAI`. You can use 
the `docker run` command to start the container. 

We need the directory above so that we can mount sample notebooks and dataset
into the container using docker volumes.

### Increase the RAM available to Docker container

Increase the Memory available to docker containers from the docker UI as shown below


![docker-settings][docker-settings]

[docker-settings]: ./images/docker_memory_settings.png 

### Set TransmogrifaiPATH

```bash
export TransmogrifaiPATH=<TransmogrifAI installation dir>
```

### Run the beakerx Container 

```bash
docker run -p 8888:8888 -v $TransmogrifaiPATH/helloworld/notebooks:/home/beakerx/helloworld-notebooks \
-v $TransmogrifaiPATH/helloworld:/home/beakerx/helloworld --name transmogrifai-container beakerx/beakerx
```

This will download the image (which takes a few minutes first time) and start the container. It will also publish the url
and the token to access the container

Sample url is shown below.

```
http://localhost:8888/?token=<sometoken>
```

On opening the image in the browser you will notice that in the home page

![notebook_home][notebook_home]

helloworld-notebooks mounted folder  (`/home/beakerx/helloworld-notebooks`)is where all our samples are located.

![helloworld_notebooks][helloworld_notebooks]

[notebook_home]: ./images/notebook_home.png 
[helloworld_notebooks]: ./images/helloworld_notebooks.png 


### Sample Notebooks

Following notebooks are currently available

#### OpTitanicSimple

[OpTitanicSimple.ipynb](http://localhost:8888/notebooks/helloworld-notebooks/OpTitanicSimple.ipynb)

![op_titanic][op_titanic]

#### OpIris

[OpIris.ipynb](http://localhost:8888/notebooks/helloworld-notebooks/OpIris.ipynb)

![op_iris][op_iris]

[op_titanic]: ./images/op_titanic.png 
[op_iris]: ./images/op_iris.png 