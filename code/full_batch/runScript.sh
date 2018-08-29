KERAS_BACKEND=theano python training.py -model Dense -data pubmed -dim 40 -nlayers 10 -nmean 3 -y 0 -sample "../data/pubmedSample20P.json"

mv log/pubmed.txt log/pubmed20P.txt

KERAS_BACKEND=theano python training.py -model Dense -data pubmed -dim 40 -nlayers 10 -nmean 3 -y 0 -sample "../data/pubmedSample40P.json"

mv log/pubmed.txt log/pubmed40P.txt

KERAS_BACKEND=theano python training.py -model Dense -data pubmed -dim 40 -nlayers 10 -nmean 3 -y 0 -sample "../data/pubmedSample60P.json"

mv log/pubmed.txt log/pubmed60P.txt

KERAS_BACKEND=theano python training.py -model Dense -data pubmed -dim 40 -nlayers 10 -nmean 3 -y 0 -sample "../data/pubmedSample80P.json"

mv log/pubmed.txt log/pubmed80P.txt

KERAS_BACKEND=theano python training.py -model Dense -data pubmed -dim 40 -nlayers 10 -nmean 3 -y 0 -sample "../data/pubmedSample100P.json"

mv log/pubmed.txt log/pubmed100P.txt
