KERAS_BACKEND=theano python trainingC.py -model Dense -data corporate -dim 40 -nlayers 10 -nmean 540 -y 0 -sample "../data/corporateSample5P.json"
mv log/pubmed.txt log/corporate3Pad.txt

KERAS_BACKEND=theano python trainingC.py -model Dense -data corporate -dim 40 -nlayers 10 -nmean 540 -y 0 -sample "../data/corporateSample10P.json"
mv log/pubmed.txt log/corporate6Pad.txt

KERAS_BACKEND=theano python trainingC.py -model Dense -data corporate -dim 40 -nlayers 10 -nmean 540 -y 0 -sample "../data/corporateSample20P.json"
mv log/pubmed.txt log/corporate12Pad.txt

KERAS_BACKEND=theano python trainingC.py -model Dense -data corporate -dim 40 -nlayers 10 -nmean 540 -y 0 -sample "../data/corporateSample60P.json"
mv log/pubmed.txt log/corporate36Pad.txt

KERAS_BACKEND=theano python trainingC.py -model Dense -data corporate -dim 40 -nlayers 10 -nmean 540 -y 0 -sample "../data/corporateSample80P.json"
mv log/pubmed.txt log/corporate48Pad.txt

KERAS_BACKEND=theano python trainingC.py -model Dense -data corporate -dim 40 -nlayers 10 -nmean 540 -y 0 -sample "../data/corporateSample100P.json"
mv log/pubmed.txt log/corporate60Pad.txt

