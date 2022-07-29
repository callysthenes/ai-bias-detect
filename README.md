# ai-bias-detect

![image](https://user-images.githubusercontent.com/91435423/181649384-47ccf79c-f0bf-4d5e-8c3b-dacd0a14a4cb.png)

Welcome to Group A's group work. We would like to introduce an AI bias detection, a compilation and comparison between two powerful AI libraries for bias detection.

![image](https://user-images.githubusercontent.com/91435423/181649485-77a397be-6cb2-41c7-b2e6-60e37bd28f2b.png)

This repository has a comparison between Google's What If Bias tool and fairdetect by Ryan Daher with some added improvements.
The whole analysis is displayed in two Jupyter notebooks for added convenience.


![image](https://user-images.githubusercontent.com/91435423/181716546-52c48452-123b-4e0d-aca3-687426595016.png)

## This repo includes:

- A class file to deal with a Pandas DF and convert it into a DB
- A class file containing all fairdetect's original functions with added improvements for both the bias detection, as well as EDI enhancements
- Original csv files where a separate analysis has been conducted
- Picture files used in the python notebook
- A DB file from which the program reads
- Folders with previous versions, the original functions not converted in a class, and some json files to use together with the Whatif tool



![image](https://user-images.githubusercontent.com/91435423/181716386-dafcce7e-858f-4b53-9e4b-fe56abbff528.png)

## How to use ai-bias-detect?

- Simply use the python notebook provided as a template
- If you want to transform your file in a db, modify the sqlalchemy file
- Replace it with a file you want to analyze for bias unfairness
- Make sure you have identified your sensitive group
- Use the docstrings to find help on how to use the different methods
- Fairdetect will provide SHAP graphs allowing to see black box models with the help of chi square tests
- Google WhatIf provides a graphical way to see bias by plotting values in real time
-  Feature importance is also covered with the help of a third library


![image](https://user-images.githubusercontent.com/91435423/181649718-ed262cec-d30e-4bdd-a9ba-423dce06281d.png)

## Some noticed bugs:

- In Apple hardware, sometimes the Kernel dies without explanation
- Pandas Profiling Tool does not seem to work consistently accross different machines running other versions of Jupyter or Python
- Google's Whatif Tool does not receive correctly the API key to unlock extra functionalities like mitigation strategies

## To improve: 

- Add persistent changes functionality to the DB after running a ML model forecasting the output of y
