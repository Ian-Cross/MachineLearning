DATA_FOLDER=./dataSets/
PROJECT_FOLDER=code
# Help pages
page?=1
test=5

all:
	@echo "*** Welcome to my learning center ***"
	@echo "There are a variety of projects in here with varying complexity"
	@echo "that I have worked on while trying to learn the world of machine learning"
	@echo "Try running \`make menu\` to see some projects"

menu: makefile
	@echo "*** Menu ***"
	@sed -n 's/^##//p' $<

## ** Scales Documentation **
## ` make scales `
## 
## 
##
.PHONY : scales
scales:
	@pipenv run python -m $(PROJECT_FOLDER).BalancedScales.scales

## ** Houses Documentation **
## ` make houses `
## Based off of the California censous data of 1990, estimate the value of a house.
## Using a couple factors: Location (Long, Lat), age of a house, total rooms, total bedrooms, total population, etc.
##
.PHONY : houses
houses:
	@pipenv run python -m $(PROJECT_FOLDER).HousePrices.src.housing

## ** Iris Documentation **
## ` make iris `
## Based off of the basic toy dataset of labeled iris flowers.
## Using a couple factors: sepal length and width, and petal length and width.
##
.PHONY : iris
iris:
	@pipenv run python $(PROJECT_FOLDER)IrisClassification/iris.py