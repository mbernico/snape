**type**: string, (default=classification)

		The type of dataset.

**n_classes**: int, (default=2)

		The number of classes (or labels) of the classification problem.   

**n_samples**: int, (default=10000)

		The number of observations

**n_features**: int, (default=100)

		The number of features

**out_path**: string, (default=“./“)

		The output system path 

**output**: string, (default=“my_dataset”)

		The output file name

**n_informative**: int, (default=30)

		The number of informative features

**n_duplicate**: int, (default=3)

		The number of perfect collinear features

**n_redundant**: int, (default=5)

		The number of multicolinear features

**n_clusters**: int, (default=4) 

		The number of gaussian clusters per class

**weights**: list of floats, (default=[0.8,0.2])

		A list of class balances

**pct_missing**: float, (default=0.01)

		The percentage of rows that should have a missing value.

**insert_dollar**: character, (default=”Yes”)

		Include a dollar sign

**insert_percent**:string, (default=”Yes”)

		Include a percent symbol

**n_categorical**: int, (default=3)

		The number of categorical variables to create

**label_list**: list of lists, (default=[[“america","asia", "euorpe"], ["monday", "tuesday", "wednesday", "thurday", "friday"], ["January","Feb","Mar","Apr","May","Jun","July", "Aug", "sept.","Oct","Nov","Dev”]])

		A list of lists, each list is the labels for one categorical variable.
