KERAS_BACKEND=theano python trainingD.py -model Dense -data debate -dim 40 -nlayers 10 -nmean 2 -y 0 -sample "../data/debateSample5P.json"
mv log/pubmed.txt log/debate3Pad.txt


KERAS_BACKEND=theano python trainingD.py -model Dense -data debate -dim 40 -nlayers 10 -nmean 2 -y 0 -sample "../data/debateSample10P.json"
mv log/pubmed.txt log/debate6Pad.txt

KERAS_BACKEND=theano python trainingD.py -model Dense -data debate -dim 40 -nlayers 10 -nmean 2 -y 0 -sample "../data/debateSample20P.json"
mv log/pubmed.txt log/debate12Pad.txt

KERAS_BACKEND=theano python trainingD.py -model Dense -data debate -dim 40 -nlayers 10 -nmean 2 -y 0 -sample "../data/debateSample40P.json"
mv log/pubmed.txt log/debate24Pad.txt


KERAS_BACKEND=theano python trainingD.py -model Dense -data debate -dim 40 -nlayers 10 -nmean 2 -y 0 -sample "../data/debateSample60P.json"
mv log/pubmed.txt log/debate36Pad.txt


KERAS_BACKEND=theano python trainingD.py -model Dense -data debate -dim 40 -nlayers 10 -nmean 2 -y 0 -sample "../data/debateSample80P.json"
mv log/pubmed.txt log/debate48Pad.txt


KERAS_BACKEND=theano python trainingD.py -model Dense -data debate -dim 40 -nlayers 10 -nmean 2 -y 0 -sample "../data/debateSample100P.json"
mv log/pubmed.txt log/debate60Pad.txt
