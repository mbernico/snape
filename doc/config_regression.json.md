**type**: string, (default=regression)

		The type of dataset.

**n_samples**: int, (default=10000)

		The number of observations.

**n_features**: int, (default=100)

		The number of features.

**out_path**: string, (default=“./“)

		The output system path.

**output**: string, (default=“my_dataset”)

		The output file name.

**n_informative**: int, (default=30)

		The number of informative features.

**n_targets**: int, (default=1)

		The number of regression targets, i.e., the dimension of the y output vector associated with a sample. By default, the output is a scalar.

**effective_rank**: int, (default=1)

		The approximate number of singular vectors required to explain data.

**tail_strength**: float, (default=0.5)

		The relative importance of the fat noisy tail of the singular values profile.

**noise**: float, (default=0.0)

		The standard deviation of the gaussian noise applied to the output.

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

**shuffle**:bool, (default=true)

		Shuffle samples and the features.

