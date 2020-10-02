DATA_FOLDER=./dataSets/
PROJECT_FOLDER=./code/
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

## Scales Documentation
.PHONY : scales
scales:
	python3 ./code/scales.py

## Houses Documentation
.PHONY : houses
houses:
	python3 ./code/houses.py

## Flower Documentation
.PHONY : flower
flower:
	python3 ./code/premade_estimator.py

## Iris Documentation
.PHONY : iris
iris:
	python3 ./code/iris2.py