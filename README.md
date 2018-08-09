## To Run

### Make sure that homebrew is installed
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

### Ensure that python 3.6.5_1 is installed not python 2.6.7 or 3.7
* check with ` python --verison `

### If the python version is not correct
* run ` brew unlink python `
* run ` brew install https://raw.githubusercontent.com/Homebrew/homebrew-core/f2a764ef944b1080be64bd88dca9a1d80130c558/Formula/python.rb `

### Package Installation
* pip3 install --upgrade virtualenv
* run ` virtualenv --system-site-packages -p python3 virtualPy
* run ` source virtualPy/bin/activate `
* run ` pip3 install pandas `
* run ` pip3 install matplotlib `
* run ` pip3 install sklearn `
* run ` pip3 install scipy `
* run ` make $projectName `