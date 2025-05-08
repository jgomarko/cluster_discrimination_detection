# Dataset Information

This directory is used for storing datasets. The example code will automatically download the Adult dataset if it's not already present.

## Adult Dataset

The Adult dataset (also known as "Census Income" dataset) contains demographic information extracted from the 1994 Census database. The task is to predict whether income exceeds $50K/year based on census data.

### Files
- `adult.data`: Training set
- `adult.test`: Test set (not used in the example)
- `adult.names`: Dataset documentation (not downloaded by default)

### Features
- **age**: continuous
- **workclass**: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked
- **fnlwgt**: continuous (final weight - number of people the census believes the entry represents)
- **education**: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool
- **education-num**: continuous
- **marital-status**: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse
- **occupation**: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces
- **relationship**: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried
- **race**: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black
- **sex**: Female, Male
- **capital-gain**: continuous
- **capital-loss**: continuous
- **hours-per-week**: continuous
- **native-country**: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands

### Target Variable
- **income**: >50K, <=50K

## Source
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.