KERAS_BACKEND=theano python trainingS.py -model Dense -data social -dim 40 -nlayers 10 -nmean 20 -y 0 -sample "../data/socialSample5P.json"
mv log/pubmed.txt log/social3Pad.txt

KERAS_BACKEND=theano python trainingS.py -model Dense -data social -dim 40 -nlayers 10 -nmean 20 -y 0 -sample "../data/socialSample10P.json"
mv log/pubmed.txt log/social6Pad.txt

KERAS_BACKEND=theano python trainingS.py -model Dense -data social -dim 40 -nlayers 10 -nmean 20 -y 0 -sample "../data/socialSample20P.json"
mv log/pubmed.txt log/social12Pad.txt

KERAS_BACKEND=theano python trainingS.py -model Dense -data social -dim 40 -nlayers 10 -nmean 20 -y 0 -sample "../data/socialSample60P.json"
mv log/pubmed.txt log/social36Pad.txt

KERAS_BACKEND=theano python trainingS.py -model Dense -data social -dim 40 -nlayers 10 -nmean 20 -y 0 -sample "../data/socialSample80P.json"
mv log/pubmed.txt log/social48Pad.txt

KERAS_BACKEND=theano python trainingS.py -model Dense -data social -dim 40 -nlayers 10 -nmean 20 -y 0 -sample "../data/socialSample100P.json"
mv log/pubmed.txt log/social60Pad.txt

